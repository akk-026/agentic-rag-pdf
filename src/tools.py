import ast
import operator as op
from typing import Any

from duckduckgo_search import DDGS
from langchain_core.tools import tool


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.Num):  # compatibility
            return node.n

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            operator_type = type(node.op)
            if operator_type not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_OPERATORS[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            operator_type = type(node.op)
            if operator_type not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_OPERATORS[operator_type](operand)

        raise ValueError("Unsupported expression.")

    return _eval(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        result = _safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web and return top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Web search error: {e}"

    if not results:
        return "No web results found."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "No title")
        href = item.get("href", "")
        body = item.get("body", "")
        lines.append(f"{i}. {title}\n{href}\n{body}")

    return "\n\n".join(lines)