from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from itertools import cycle
import os
import re
import sys
import types
from pathlib import Path

try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None  # type: ignore[assignment]

import art
from art.dev import InternalModelConfig
from art.local import LocalBackend
from art.pipeline_trainer import PipelineTrainer


DEFAULT_BASE_MODEL = os.environ.get("ART_BASE_MODEL", "Qwen/Qwen3-0.6B")
PROJECT = os.environ.get("ART_PROJECT", "art-local-tool-smoke")
MODEL_PREFIX = os.environ.get("ART_MODEL_PREFIX", "qwen3-0.6b-local-tool-smoke")
ROLLOUTS_PER_SCENARIO = int(os.environ.get("ART_ROLLOUTS_PER_SCENARIO", "8"))
MAX_TOKENS = int(os.environ.get("ART_MAX_TOKENS", "48"))
MAX_STEPS = int(os.environ.get("ART_MAX_STEPS", "2"))
ROLLOUT_TEMPERATURE = float(os.environ.get("ART_ROLLOUT_TEMPERATURE", "1.0"))
EVAL_TEMPERATURE = float(os.environ.get("ART_EVAL_TEMPERATURE", "0.2"))
MIN_BATCH_SIZE = int(os.environ.get("ART_MIN_BATCH_SIZE", "2"))
DISCARD_QUEUE_MULTIPLIER = int(os.environ.get("ART_DISCARD_QUEUE_MULTIPLIER", "500"))
ART_PATH = os.environ.get(
    "ART_ART_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".art_local"),
)

RESPONSE_RE = re.compile(
    r"TOOL\s*[:=]\s*(?P<tool>[a-zA-Z_]+).*?"
    r"RESULT\s*[:=]\s*(?P<result>-?\d+)",
    re.IGNORECASE | re.DOTALL,
)

TOOLS: dict[str, callable] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: a if a >= b else b,
}


def resolve_base_model(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name

    if model_name == "Qwen/Qwen3-0.6B":
        snapshot_root = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
        if snapshot_root.exists():
            snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
            if snapshots:
                return str(snapshots[-1])
    return model_name


BASE_MODEL = resolve_base_model(DEFAULT_BASE_MODEL)


def build_scenarios() -> list[dict[str, object]]:
    raw_tasks = [
        ("add", 7, 5),
        ("add", 14, 9),
        ("sub", 12, 8),
        ("sub", 19, 7),
        ("mul", 3, 4),
        ("mul", 6, 5),
        ("max", 11, 4),
        ("max", 8, 13),
        ("add", 21, 17),
        ("sub", 25, 9),
        ("mul", 7, 3),
        ("max", 16, 16),
    ]
    templates = [
        (
            "Available tools: add, sub, mul, max.\n"
            "Task: compute {tool} on inputs {a} and {b}.\n"
            "Reply exactly as: TOOL=<tool> RESULT=<n>"
        ),
        (
            "You may call one tool from [add, sub, mul, max].\n"
            "Use the requested tool `{tool}` with operands {a} and {b}.\n"
            "Output only: TOOL=<tool> RESULT=<n>"
        ),
        (
            "Pick the correct tool and show its arguments.\n"
            "Required operation: {tool}({a}, {b}).\n"
            "Strict format: TOOL=<tool> RESULT=<n>"
        ),
    ]

    scenarios: list[dict[str, object]] = []
    for index, (tool, a, b) in enumerate(raw_tasks):
        prompt = templates[index % len(templates)].format(tool=tool, a=a, b=b)
        scenarios.append(
            {
                "prompt": prompt,
                "tool": tool,
                "a": a,
                "b": b,
                "result": TOOLS[tool](a, b),
            }
        )
    return scenarios


def parse_response(text: str) -> dict[str, object] | None:
    match = RESPONSE_RE.search(text)
    tool: str | None = None
    result: int | None = None

    if match:
        tool = match.group("tool").lower()
        result = int(match.group("result"))
    else:
        lowered = text.lower()
        for candidate in TOOLS:
            if re.search(rf"\b{candidate}\b", lowered):
                tool = candidate
                break
        ints = re.findall(r"-?\d+", text)
        if ints:
            result = int(ints[-1])

    if tool is None and result is None:
        return None
    return {"tool": tool, "result": result}


def reward_for_answer(text: str, scenario: dict[str, object]) -> float:
    parsed = parse_response(text)
    if parsed is None:
        return -0.2

    reward = 0.0
    expected_tool = scenario["tool"]
    expected_result = scenario["result"]

    reward += 0.05
    if parsed["tool"] == expected_tool:
        reward += 0.45
    elif parsed["tool"] is not None:
        reward -= 0.05

    if parsed["result"] == expected_result:
        reward += 0.45
    elif parsed["result"] is not None:
        distance = abs(int(parsed["result"]) - int(expected_result))
        reward += max(0.0, 0.2 - 0.02 * distance)

    if parsed["tool"] == expected_tool and parsed["result"] == expected_result:
        reward += 0.1

    return max(-0.2, min(reward, 1.1))


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


async def eval_fn(
    model: art.TrainableModel,
    step: int,
    _config: None,
    *,
    scenarios: list[dict[str, object]],
) -> list[art.Trajectory]:
    trajectories: list[art.Trajectory] = []
    client = model.openai_client()
    inference_name = model.get_inference_name(step)

    for scenario in scenarios[:4]:
        messages: art.Messages = [{"role": "user", "content": str(scenario["prompt"])}]
        response = await client.chat.completions.create(
            messages=messages,
            model=inference_name,
            max_tokens=MAX_TOKENS,
            n=1,
            temperature=EVAL_TEMPERATURE,
        )
        choice = response.choices[0]
        reward = reward_for_answer(choice.message.content or "", scenario)
        trajectories.append(
            art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)
        )
    return trajectories


async def rollout_fn(
    model: art.TrainableModel,
    scenario: dict[str, object],
    _config: None,
) -> art.TrajectoryGroup:
    messages: art.Messages = [{"role": "user", "content": str(scenario["prompt"])}]
    response = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model.get_inference_name(),
        max_tokens=MAX_TOKENS,
        n=ROLLOUTS_PER_SCENARIO,
        temperature=ROLLOUT_TEMPERATURE,
    )
    return art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=reward_for_answer(choice.message.content or "", scenario),
            )
            for choice in response.choices
        ]
    )


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

    scenarios_list = build_scenarios()
    print(f"[ART] backend path: {ART_PATH}")
    print(f"[ART] base model: {BASE_MODEL}")
    print("[ART] dedicated mode: trainer_gpu_ids=[0], inference_gpu_ids=[1]")
    print(
        f"[ART] tool-use task: scenarios={len(scenarios_list)}, "
        f"rollout_temperature={ROLLOUT_TEMPERATURE}, eval_temperature={EVAL_TEMPERATURE}"
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

    scenarios = cycle(scenarios_list)
    eval_callback = partial(eval_fn, scenarios=scenarios_list)

    trainer = PipelineTrainer(
        model=model,
        backend=backend,
        rollout_fn=rollout_fn,
        scenarios=scenarios,
        config=None,
        learning_rate=5e-6,
        loss_fn="cispo",
        eval_fn=eval_callback,
        min_batch_size=MIN_BATCH_SIZE,
        discard_queue_multiplier=DISCARD_QUEUE_MULTIPLIER,
        max_steps=MAX_STEPS,
        eval_every_n_steps=1,
        eval_at_start=False,
        save_checkpoint=False,
    )

    print(
        f"[ART] training start: steps={MAX_STEPS}, "
        f"rollouts_per_scenario={ROLLOUTS_PER_SCENARIO}, max_tokens={MAX_TOKENS}, "
        f"min_batch_size={MIN_BATCH_SIZE}"
    )
    try:
        await trainer.train()
    finally:
        await backend.close()
        print("[ART] backend closed")


if __name__ == "__main__":
    asyncio.run(main())
