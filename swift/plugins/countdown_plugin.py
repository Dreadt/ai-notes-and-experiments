import ast
import operator
import re
from collections import Counter
from typing import List

from swift.rewards import ORM, orms


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _extract_expression(text: str) -> str:
    match = re.findall(r"####\s*([^\n\r]+)", text)
    if match:
        return match[-1].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _safe_eval_expr(expr: str):
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("unsupported expression")

    return _eval(tree)


def _extract_numbers(expr: str):
    return [int(x) for x in re.findall(r"\d+", expr)]


class CountdownCorrect(ORM):
    def __call__(self, completions, nums, target, **kwargs) -> List[float]:
        rewards = []
        for completion, allowed_nums, gold_target in zip(completions, nums, target):
            expr = _extract_expression(completion)
            if not expr:
                rewards.append(0.0)
                continue
            try:
                used_nums = _extract_numbers(expr)
                if Counter(used_nums) != Counter(int(x) for x in allowed_nums):
                    rewards.append(0.0)
                    continue
                value = _safe_eval_expr(expr)
                rewards.append(1.0 if abs(value - float(gold_target)) < 1e-6 else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards


class CountdownFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            expr = _extract_expression(completion)
            ok = bool(expr) and bool(re.fullmatch(r"[0-9+\-*/().\s]+", expr))
            rewards.append(1.0 if ok else 0.0)
        return rewards


orms["countdown_correct"] = CountdownCorrect
orms["countdown_format"] = CountdownFormat
