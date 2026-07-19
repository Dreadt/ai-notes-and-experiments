from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import cycle
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Any

try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None  # type: ignore[assignment]

import art
from art.dev import InternalModelConfig
from art.local import LocalBackend
from art.mcp.types import MCPTool
from art.pipeline_trainer import PipelineTrainer
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import AsyncOpenAI


DEFAULT_BASE_MODEL = os.environ.get("ART_BASE_MODEL", "Qwen/Qwen3-0.6B")
PROJECT = os.environ.get("ART_PROJECT", "art-local-mcp-rl")
MODEL_PREFIX = os.environ.get(
    "ART_MODEL_PREFIX",
    f"{DEFAULT_BASE_MODEL.split('/')[-1].lower().replace('.', 'p')}-local-mcp-rl",
)
ROLLOUTS_PER_SCENARIO = int(os.environ.get("ART_ROLLOUTS_PER_SCENARIO", "8"))
MAX_TOKENS = int(os.environ.get("ART_MAX_TOKENS", "64"))
MAX_STEPS = int(os.environ.get("ART_MAX_STEPS", "2"))
ROLLOUT_TEMPERATURE = float(os.environ.get("ART_ROLLOUT_TEMPERATURE", "1.0"))
EVAL_TEMPERATURE = float(os.environ.get("ART_EVAL_TEMPERATURE", "0.2"))
EVAL_AT_START = os.environ.get("ART_EVAL_AT_START", "0") == "1"
MIN_BATCH_SIZE = int(os.environ.get("ART_MIN_BATCH_SIZE", "2"))
DISCARD_QUEUE_MULTIPLIER = int(os.environ.get("ART_DISCARD_QUEUE_MULTIPLIER", "500"))
ART_PATH = os.environ.get(
    "ART_ART_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".art_local"),
)
SCENARIO_SOURCE = os.environ.get("ART_SCENARIO_SOURCE", "manual").strip().lower()
SYNTHETIC_SCENARIO_COUNT = int(os.environ.get("ART_SYNTHETIC_SCENARIO_COUNT", "24"))
SCENARIO_GENERATOR_BASE_URL = os.environ.get("ART_SCENARIO_GENERATOR_BASE_URL", "").strip()
SCENARIO_GENERATOR_API_KEY = os.environ.get("ART_SCENARIO_GENERATOR_API_KEY", "").strip()
SCENARIO_GENERATOR_MODEL = os.environ.get("ART_SCENARIO_GENERATOR_MODEL", "").strip()
SERVER_PATH = (
    Path(os.environ.get("ART_MCP_SERVER_PATH", ""))
    if os.environ.get("ART_MCP_SERVER_PATH")
    else Path(__file__).resolve().parents[1] / "mcp_servers" / "arithmetic_server.py"
)

RESPONSE_RE = re.compile(
    r"TOOL\s*[:=]\s*(?P<tool>[a-zA-Z0-9_]+).*?"
    r"RESULT\s*[:=]\s*(?P<result>-?\d+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class Scenario:
    prompt: str
    tool: str
    a: int
    b: int
    expected_result: int
    difficulty: int


def resolve_base_model(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name

    def is_complete_snapshot(path: Path) -> bool:
        return any(path.glob("model-*.safetensors")) or (path / "model.safetensors").exists()

    snapshot_roots = {
        "Qwen/Qwen3-0.6B": [
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots",
            "/root/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots",
        ],
        "Qwen/Qwen3-1.7B": [
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots",
            "/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots",
        ],
        "Qwen/Qwen3-4B": [
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots",
            "/root/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots",
        ],
    }
    for snapshot_root_value in snapshot_roots.get(model_name, []):
        snapshot_root = Path(snapshot_root_value)
        if not snapshot_root.exists():
            continue
        snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
        for snapshot in reversed(snapshots):
            if is_complete_snapshot(snapshot):
                return str(snapshot)
    return model_name


BASE_MODEL = resolve_base_model(DEFAULT_BASE_MODEL)


def install_model_support_shim() -> None:
    module_name = "art.megatron.model_support"
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)

    def default_target_modules_for_model(
        base_model: str,
        allow_unvalidated_arch: bool = True,
    ) -> list[str]:
        del allow_unvalidated_arch
        lowered = base_model.lower()
        if "qwen" in lowered:
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    shim.default_target_modules_for_model = default_target_modules_for_model
    sys.modules[module_name] = shim


def parse_response(text: str) -> dict[str, int | str] | None:
    match = RESPONSE_RE.search(text)
    tool: str | None = None
    result: int | None = None

    if match:
        tool = match.group("tool").lower()
        result = int(match.group("result"))
    else:
        lowered = text.lower()
        ints = re.findall(r"-?\d+", text)
        for candidate in ("add", "sub", "mul", "max2"):
            if re.search(rf"\b{candidate}\b", lowered):
                tool = candidate
                break
        if ints:
            result = int(ints[-1])

    if tool is None and result is None:
        return None
    return {"tool": tool, "result": result}


def extract_scalar_tool_result(raw_result: object) -> int | None:
    if hasattr(raw_result, "structuredContent"):
        structured = getattr(raw_result, "structuredContent")
        if isinstance(structured, dict):
            for key in ("result", "value"):
                value = structured.get(key)
                if isinstance(value, int):
                    return value

    if hasattr(raw_result, "content"):
        content = getattr(raw_result, "content")
        if isinstance(content, Sequence):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text = text.strip()
                    if text.lstrip("-").isdigit():
                        return int(text)
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        for key in ("result", "value"):
                            value = payload.get(key)
                            if isinstance(value, int):
                                return value
    return None


def build_scenarios(discovered_tools: list[MCPTool], tool_results: dict[str, int]) -> list[Scenario]:
    available_tools = {tool.name for tool in discovered_tools}
    raw_tasks = [
        ("add", 7, 5, 1),
        ("add", 14, 9, 1),
        ("sub", 19, 7, 2),
        ("sub", 12, 8, 1),
        ("mul", 6, 5, 2),
        ("mul", 7, 3, 2),
        ("max2", 11, 4, 1),
        ("max2", 8, 13, 1),
        ("add", 21, 17, 2),
        ("sub", 25, 9, 2),
        ("mul", 3, 4, 1),
        ("max2", 16, 16, 1),
    ]
    templates = [
        (
            "You are connected to an MCP server exposing these tools: {tools}.\n"
            "Task: use the correct MCP tool to solve `{tool}({a}, {b})`.\n"
            "Reply exactly as: TOOL=<tool> RESULT=<n>"
        ),
        (
            "Available MCP tools: {tools}.\n"
            "Choose the best tool for operands {a} and {b} to complete `{tool}`.\n"
            "Output only: TOOL=<tool> RESULT=<n>"
        ),
        (
            "MCP tool inventory: {tools}.\n"
            "Determine the correct tool invocation for {tool} on inputs {a} and {b}.\n"
            "Strict format: TOOL=<tool> RESULT=<n>"
        ),
    ]

    scenarios: list[Scenario] = []
    tools_string = ", ".join(sorted(available_tools))
    for index, (tool, a, b, difficulty) in enumerate(raw_tasks):
        if tool not in available_tools:
            continue
        prompt = templates[index % len(templates)].format(
            tools=tools_string,
            tool=tool,
            a=a,
            b=b,
        )
        scenarios.append(
            Scenario(
                prompt=prompt,
                tool=tool,
                a=a,
                b=b,
                expected_result=tool_results[f"{tool}:{a}:{b}"],
                difficulty=difficulty,
            )
        )
    return scenarios


def _make_generator_client() -> AsyncOpenAI:
    if not SCENARIO_GENERATOR_API_KEY:
        raise RuntimeError("ART_SCENARIO_GENERATOR_API_KEY is required when ART_SCENARIO_SOURCE=llm")
    if not SCENARIO_GENERATOR_MODEL:
        raise RuntimeError("ART_SCENARIO_GENERATOR_MODEL is required when ART_SCENARIO_SOURCE=llm")
    kwargs: dict[str, Any] = {"api_key": SCENARIO_GENERATOR_API_KEY}
    if SCENARIO_GENERATOR_BASE_URL:
        kwargs["base_url"] = SCENARIO_GENERATOR_BASE_URL
    return AsyncOpenAI(**kwargs)


def _coerce_generated_scenarios(payload: Any, available_tools: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("scenarios", [])
    if not isinstance(payload, list):
        raise RuntimeError("Scenario generator did not return a JSON list")

    normalized: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", "")).strip().lower()
        if tool not in available_tools:
            continue
        try:
            a = int(item["a"])
            b = int(item["b"])
        except Exception:
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            prompt = (
                f"You are connected to an MCP server exposing these tools: {', '.join(sorted(available_tools))}.\n"
                f"Use the correct MCP tool to solve `{tool}({a}, {b})`.\n"
                "Reply exactly as: TOOL=<tool> RESULT=<n>"
            )
        difficulty = int(item.get("difficulty", 1))
        normalized.append(
            {
                "prompt": prompt,
                "tool": tool,
                "a": a,
                "b": b,
                "difficulty": max(1, min(difficulty, 3)),
            }
        )
    return normalized


async def generate_scenarios_with_llm(
    discovered_tools: list[MCPTool],
    session: ClientSession,
) -> list[Scenario]:
    available_tools = {tool.name for tool in discovered_tools}
    client = _make_generator_client()
    tools_string = ", ".join(sorted(available_tools))
    prompt = f"""
You are generating synthetic MCP-RL training scenarios.

Available MCP tools: {tools_string}

Return strictly valid JSON as a list. Each item must have:
- prompt: string
- tool: one of [{tools_string}]
- a: integer
- b: integer
- difficulty: integer from 1 to 3

Requirements:
- Generate exactly {SYNTHETIC_SCENARIO_COUNT} diverse arithmetic tasks.
- Prompts must mention the available MCP tools and ask the model to solve a task by using the correct tool.
- Keep prompts short and practical.
- Use only integer inputs between 0 and 40.
- Ensure the target tool matches the intended operation.
- Do not include markdown fences.
""".strip()

    response = await client.chat.completions.create(
        model=SCENARIO_GENERATOR_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You generate compact JSON datasets for tool-use reinforcement learning.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    generated_items = _coerce_generated_scenarios(parsed, available_tools)
    if len(generated_items) < max(8, SYNTHETIC_SCENARIO_COUNT // 2):
        raise RuntimeError(
            f"Scenario generator returned too few valid scenarios: {len(generated_items)}"
        )

    scenarios: list[Scenario] = []
    for item in generated_items:
        result = await session.call_tool(item["tool"], {"a": item["a"], "b": item["b"]})
        scalar = extract_scalar_tool_result(result)
        if scalar is None:
            continue
        scenarios.append(
            Scenario(
                prompt=item["prompt"],
                tool=item["tool"],
                a=item["a"],
                b=item["b"],
                expected_result=scalar,
                difficulty=item["difficulty"],
            )
        )
    if not scenarios:
        raise RuntimeError("Scenario generator did not yield any executable MCP scenarios")
    return scenarios


async def score_answer(
    session: ClientSession,
    text: str,
    scenario: Scenario,
    allowed_tools: set[str],
) -> float:
    parsed = parse_response(text)
    if parsed is None:
        return -0.2

    reward = 0.05
    tool_name = str(parsed["tool"]) if parsed["tool"] is not None else ""
    if tool_name in allowed_tools:
        reward += 0.1
    else:
        return -0.15

    if tool_name == scenario.tool:
        reward += 0.25

    try:
        tool_result = await session.call_tool(
            tool_name,
            {"a": scenario.a, "b": scenario.b},
        )
        actual_result = extract_scalar_tool_result(tool_result)
    except Exception:
        return max(-0.2, reward - 0.1)

    if actual_result is None:
        return max(-0.2, reward - 0.05)

    parsed_result = parsed["result"]
    if isinstance(parsed_result, int) and parsed_result == actual_result:
        reward += 0.2

    if tool_name == scenario.tool and actual_result == scenario.expected_result:
        reward += 0.25

    if isinstance(parsed_result, int) and parsed_result == scenario.expected_result:
        reward += 0.2
    elif isinstance(parsed_result, int):
        reward += max(0.0, 0.1 - 0.02 * abs(parsed_result - scenario.expected_result))

    return max(-0.2, min(reward, 1.1))


async def eval_fn(
    model: art.TrainableModel,
    step: int,
    _config: None,
    *,
    scenarios: list[Scenario],
    mcp_session: ClientSession,
    allowed_tools: set[str],
) -> list[art.Trajectory]:
    trajectories: list[art.Trajectory] = []
    client = model.openai_client()
    inference_name = model.get_inference_name(step)

    for scenario in scenarios[:4]:
        messages: art.Messages = [{"role": "user", "content": scenario.prompt}]
        response = await client.chat.completions.create(
            messages=messages,
            model=inference_name,
            max_tokens=MAX_TOKENS,
            n=1,
            temperature=EVAL_TEMPERATURE,
        )
        choice = response.choices[0]
        reward = await score_answer(
            mcp_session,
            choice.message.content or "",
            scenario,
            allowed_tools,
        )
        trajectories.append(
            art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)
        )
    return trajectories


async def rollout_fn(
    model: art.TrainableModel,
    scenario: Scenario,
    _config: None,
    *,
    mcp_session: ClientSession,
    allowed_tools: set[str],
) -> art.TrajectoryGroup:
    messages: art.Messages = [{"role": "user", "content": scenario.prompt}]
    response = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model.get_inference_name(),
        max_tokens=MAX_TOKENS,
        n=ROLLOUTS_PER_SCENARIO,
        temperature=ROLLOUT_TEMPERATURE,
    )
    trajectories = []
    for choice in response.choices:
        reward = await score_answer(
            mcp_session,
            choice.message.content or "",
            scenario,
            allowed_tools,
        )
        trajectories.append(
            art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)
        )
    return art.TrajectoryGroup(trajectories)


async def collect_tools_and_scenarios(
    session: ClientSession,
) -> tuple[list[MCPTool], list[Scenario]]:
    listed = await session.list_tools()
    tools = [
        MCPTool(
            name=tool.name,
            description=getattr(tool, "description", "") or "",
            parameters=getattr(tool, "inputSchema", {}) or {},
        )
        for tool in listed.tools
    ]

    tool_results: dict[str, int] = {}
    probe_inputs = [
        ("add", 7, 5),
        ("add", 14, 9),
        ("sub", 19, 7),
        ("sub", 12, 8),
        ("mul", 6, 5),
        ("mul", 7, 3),
        ("max2", 11, 4),
        ("max2", 8, 13),
        ("add", 21, 17),
        ("sub", 25, 9),
        ("mul", 3, 4),
        ("max2", 16, 16),
    ]
    available = {tool.name for tool in tools}
    for tool_name, a, b in probe_inputs:
        if tool_name not in available:
            continue
        result = await session.call_tool(tool_name, {"a": a, "b": b})
        scalar = extract_scalar_tool_result(result)
        if scalar is None:
            raise RuntimeError(f"Failed to parse tool result for {tool_name}({a}, {b})")
        tool_results[f"{tool_name}:{a}:{b}"] = scalar

    if SCENARIO_SOURCE == "llm":
        scenarios = await generate_scenarios_with_llm(tools, session)
    else:
        scenarios = build_scenarios(tools, tool_results)
    if not scenarios:
        raise RuntimeError("No scenarios were built from discovered MCP tools")
    return tools, scenarios


async def main() -> None:
    install_model_support_shim()
    model_name = f"{MODEL_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    internal_config = InternalModelConfig(
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
        init_args={
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": True,
            "use_exact_model_name": True,
            "local_files_only": True,
        },
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(SERVER_PATH.parent.parent),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_session:
            await mcp_session.initialize()
            discovered_tools, scenario_list = await collect_tools_and_scenarios(mcp_session)
            allowed_tools = {tool.name for tool in discovered_tools}

            print(f"[ART] backend path: {ART_PATH}")
            print(f"[ART] base model: {BASE_MODEL}")
            print("[ART] dedicated mode: trainer_gpu_ids=[0], inference_gpu_ids=[1]")
            print(f"[ART] mcp server: {SERVER_PATH}")
            print(f"[ART] discovered tools: {sorted(allowed_tools)}")
            print(
                f"[ART] mcp-rl task: scenarios={len(scenario_list)}, "
                f"scenario_source={SCENARIO_SOURCE}, "
                f"rollout_temperature={ROLLOUT_TEMPERATURE}, "
                f"eval_temperature={EVAL_TEMPERATURE}"
            )

            backend = LocalBackend(path=ART_PATH)
            model = art.TrainableModel(
                name=model_name,
                project=PROJECT,
                base_model=BASE_MODEL,
                _internal_config=internal_config,
            )

            print(f"[ART] registering model: {model_name}")
            await model.register(backend)
            print("[ART] model registered")

            scenarios = cycle(scenario_list)
            eval_callback = partial(
                eval_fn,
                scenarios=scenario_list,
                mcp_session=mcp_session,
                allowed_tools=allowed_tools,
            )
            rollout_callback = partial(
                rollout_fn,
                mcp_session=mcp_session,
                allowed_tools=allowed_tools,
            )

            trainer = PipelineTrainer(
                model=model,
                backend=backend,
                rollout_fn=rollout_callback,
                scenarios=scenarios,
                config=None,
                learning_rate=5e-6,
                loss_fn="cispo",
                eval_fn=eval_callback,
                min_batch_size=MIN_BATCH_SIZE,
                discard_queue_multiplier=DISCARD_QUEUE_MULTIPLIER,
                max_steps=MAX_STEPS,
                eval_every_n_steps=1,
                eval_at_start=EVAL_AT_START,
                save_checkpoint=False,
            )

            print(
                f"[ART] training start: steps={MAX_STEPS}, "
                f"rollouts_per_scenario={ROLLOUTS_PER_SCENARIO}, "
                f"max_tokens={MAX_TOKENS}, min_batch_size={MIN_BATCH_SIZE}"
            )
            try:
                await trainer.train()
            finally:
                await backend.close()
                print("[ART] backend closed")


if __name__ == "__main__":
    asyncio.run(main())
