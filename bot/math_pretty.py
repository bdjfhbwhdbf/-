from __future__ import annotations

import re
import html
from typing import List, Tuple

from sympy import sympify
from sympy.parsing.latex import parse_latex
from sympy.printing.pretty import pretty as sympy_pretty

_SUPERSCRIPTS = str.maketrans("0123456789+-=()nkaetixy", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵏᵃᵉᵗⁱˣʸ")
_SUBSCRIPTS = str.maketrans("0123456789+-=()nkaetixy", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙₖₐₑₜᵢₓᵧ")

latex_like_pattern = re.compile(
    r"(\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{.*?\}.*?\\end\{.*?\}|\\frac\{.*?\}\{.*?\}|\\sqrt(?:\[[^\]]*\])?\{.*?\}|\\int(?:_\{.*?\})?(?:\^\{.*?\})?.*?\\,?\s*dx|\\sum(?:_\{.*?\})?(?:\^\{.*?\})?)",
    flags=re.DOTALL,
)


def to_superscript(text: str) -> str:
    return text.translate(_SUPERSCRIPTS)


def to_subscript(text: str) -> str:
    return text.translate(_SUBSCRIPTS)


def asciimath_to_inline_unicode(expr: str) -> str:
    # Very lightweight inline prettifier: a^2 -> a², x_1 -> x₁, 1/2 -> ½-like fallback
    expr = re.sub(r"\^(\d+|[a-zA-Z]+)", lambda m: to_superscript(m.group(1)), expr)
    expr = re.sub(r"_([0-9a-zA-Z]+)", lambda m: to_subscript(m.group(1)), expr)
    expr = expr.replace("sqrt", "√")
    return expr


def latex_to_pretty_block(latex: str) -> str:
    try:
        sym = parse_latex(latex)
        rendered = sympy_pretty(sym, use_unicode=True)
        return rendered
    except Exception:
        # Try to interpret as sympy ascii
        try:
            sym = sympify(latex)
            rendered = sympy_pretty(sym, use_unicode=True)
            return rendered
        except Exception:
            return asciimath_to_inline_unicode(latex)


def extract_formulas(text: str) -> List[Tuple[str, str]]:
    # Returns list of (kind, content) where kind in {"text", "latex"}
    parts: List[Tuple[str, str]] = []
    last = 0
    for m in latex_like_pattern.finditer(text):
        if m.start() > last:
            parts.append(("text", text[last:m.start()]))
        parts.append(("latex", m.group(0)))
        last = m.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def to_telegram_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Split a text into a sequence of blocks suitable for Telegram:
    - ("text", html_escaped_text) for normal paragraphs
    - ("code", html_escaped_pretty_unicode) for pretty math blocks
    """
    sequence: List[Tuple[str, str]] = []
    for kind, chunk in extract_formulas(text):
        if kind == "text":
            if chunk.strip():
                sequence.append(("text", html.escape(chunk)))
        else:
            pretty_block = latex_to_pretty_block(chunk)
            if pretty_block.strip():
                sequence.append(("code", html.escape(pretty_block)))
    return sequence