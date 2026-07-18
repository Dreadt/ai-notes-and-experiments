import re
from typing import List

from swift.rewards import ORM, orms


def _extract_answer(text: str) -> str:
    text = text[-256:] if len(text) > 256 else text
    match = re.findall(r"####\s*([^\n\r]+)", text)
    if not match:
        return ""
    return match[-1].strip().replace(",", "")


class ArithmeticExact(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, target in zip(completions, solution):
            pred = _extract_answer(completion)
            gold = _extract_answer(target)
            rewards.append(1.0 if pred and pred == gold else 0.0)
        return rewards


class ArithmeticFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            rewards.append(1.0 if bool(re.search(r"####\s*[^\n\r]+", completion)) else 0.0)
        return rewards


orms["arith_exact"] = ArithmeticExact
orms["arith_format"] = ArithmeticFormat
