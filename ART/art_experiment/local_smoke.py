from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from itertools import cycle
import os
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
PROJECT = os.environ.get("ART_PROJECT", "art-local-smoke")
MODEL_PREFIX = os.environ.get("ART_MODEL_PREFIX", "qwen3-0.6b-local-smoke")
ROLLOUTS_PER_SCENARIO = int(os.environ.get("ART_ROLLOUTS_PER_SCENARIO", "8"))
MAX_TOKENS = int(os.environ.get("ART_MAX_TOKENS", "24"))
MAX_STEPS = int(os.environ.get("ART_MAX_STEPS", "2"))
ROLLOUT_TEMPERATURE = float(os.environ.get("ART_ROLLOUT_TEMPERATURE", "1.1"))
EVAL_TEMPERATURE = float(os.environ.get("ART_EVAL_TEMPERATURE", "0.3"))
MIN_BATCH_SIZE = int(os.environ.get("ART_MIN_BATCH_SIZE", "2"))
DISCARD_QUEUE_MULTIPLIER = int(os.environ.get("ART_DISCARD_QUEUE_MULTIPLIER", "500"))
ART_PATH = os.environ.get(
    "ART_ART_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".art_local"),
)
SECRET_BITS = os.environ.get("ART_SECRET_BITS", "1100101001110011")
SECRET_LEN = len(SECRET_BITS)
SECRET_VARIANTS = [
    SECRET_BITS,
    SECRET_BITS[::-1],
    "".join("1" if bit == "0" else "0" for bit in SECRET_BITS),
    SECRET_BITS[::2] + SECRET_BITS[1::2],
]


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


def build_scenarios() -> list[dict[str, str]]:
    prompt_templates = [
        "Output exactly {length} binary digits using only 0 and 1. Do not add spaces or explanation.",
        "Respond with a {length}-character binary string. Only print the bitstring.",
        "Print one binary guess of length {length}. Allowed characters are 0 and 1 only.",
        "Your full reply must be a {length}-bit string with no punctuation.",
        "Return a binary code with exactly {length} bits. No words, no extra symbols.",
        "Generate a {length}-bit answer. The reply must contain only 0 or 1 characters.",
    ]
    scenarios: list[dict[str, str]] = []
    for template, secret in zip(cycle(prompt_templates), SECRET_VARIANTS):
        scenarios.append({"prompt": template.format(length=len(secret)), "secret": secret})
        if len(scenarios) == len(SECRET_VARIANTS):
            break
    return scenarios


def extract_bitstring(text: str) -> str:
    return "".join(ch for ch in text if ch in {"0", "1"})


def shared_prefix_len(guess: str, secret: str) -> int:
    matched = 0
    for guessed, actual in zip(guess, secret):
        if guessed != actual:
            break
        matched += 1
    return matched


def reward_for_answer(text: str, secret: str) -> float:
    guess = extract_bitstring(text)
    if not guess:
        return -0.1

    prefix_score = shared_prefix_len(guess, secret) / len(secret)
    position_matches = sum(1 for guessed, actual in zip(guess, secret) if guessed == actual)
    position_score = position_matches / len(secret)
    exact_length_bonus = 0.1 if len(guess) == len(secret) else 0.0
    valid_char_bonus = 0.05 if len(guess) >= len(secret) // 2 else 0.0
    length_penalty = abs(len(guess) - len(secret)) * 0.01
    reward = (
        0.6 * prefix_score
        + 0.3 * position_score
        + exact_length_bonus
        + valid_char_bonus
        - length_penalty
    )
    return max(-0.2, min(reward, 1.2))


def install_model_support_shim() -> None:
    """Bypass ART's eager Megatron import for a minimal local smoke run."""
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
    scenarios: list[dict[str, str]],
) -> list[art.Trajectory]:
    trajectories: list[art.Trajectory] = []
    client = model.openai_client()
    inference_name = model.get_inference_name(step)

    for scenario in scenarios:
        messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
        response = await client.chat.completions.create(
            messages=messages,
            model=inference_name,
            max_tokens=MAX_TOKENS,
            n=1,
            temperature=EVAL_TEMPERATURE,
        )
        choice = response.choices[0]
        reward = reward_for_answer(choice.message.content or "", scenario["secret"])
        trajectories.append(
            art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)
        )
    return trajectories


async def rollout_fn(
    model: art.TrainableModel,
    scenario: dict[str, str],
    _config: None,
) -> art.TrajectoryGroup:
    messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
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
                reward=reward_for_answer(choice.message.content or "", scenario["secret"]),
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

    print(f"[ART] backend path: {ART_PATH}")
    print(f"[ART] base model: {BASE_MODEL}")
    print("[ART] dedicated mode: trainer_gpu_ids=[0], inference_gpu_ids=[1]")
    print(
        f"[ART] secret bits task: len={SECRET_LEN}, "
        f"variants={len(SECRET_VARIANTS)}, rollout_temperature={ROLLOUT_TEMPERATURE}, "
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

    base_scenarios = build_scenarios()
    scenarios = cycle(base_scenarios)
    eval_callback = partial(eval_fn, scenarios=base_scenarios)

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
