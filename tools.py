"""Tool yang bisa dipanggil model.

Tool di sini sengaja dipilih yang **tidak bisa** dikerjakan model sendiri
secara andal: tanggal/waktu saat ini dan isi filesystem proyek. Tool
matematika sederhana (``add``/``multiply``) sebelumnya jarang dipakai model
karena model memang sudah bisa berhitung, sehingga loop ``llm -> tools -> llm``
hampir tidak pernah benar-benar teruji.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from log import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Batasi ukuran hasil agar tidak membanjiri konteks model.
MAX_LISTED_FILES = 60
MAX_FILE_CHARS = 4000

# Batas pangkat pada kalkulator, supaya ekspresi seperti "9 ** 9 ** 9"
# tidak membuat proses menggantung.
MAX_EXPONENT = 1000

# File yang tidak boleh dibaca tool. Tool ini dipanggil oleh model yang bisa
# saja terkena prompt injection, jadi kredensial tidak boleh ikut terkirim
# ke API provider sebagai bagian dari respons tool.
BLOCKED_FILE_NAMES = {".env", ".env.local", "credentials.json", "secrets.json"}
BLOCKED_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
BLOCKED_DIRS = {".git", ".venv", "__pycache__", ".continue"}

# Ekspresi matematika dievaluasi lewat AST whitelist, bukan eval():
# hanya literal angka, operator biner/uner, dan fungsi bernama yang dikenal.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "sqrt": lambda value: value ** 0.5,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # "9 ** 9 ** 9" secara sintaks sah tapi akan menghabiskan memori
        # dan CPU tanpa batas, jadi pangkat dibatasi.
        if type(node.op) is ast.Pow and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent must be between -{MAX_EXPONENT} "
                             f"and {MAX_EXPONENT}.")

        return _ALLOWED_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named functions are allowed.")

        name = node.func.id

        if name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unknown function: {name}")

        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")

        return _ALLOWED_FUNCTIONS[name](
            *[_safe_eval(argument) for argument in node.args]
        )

    raise ValueError("Unsupported expression.")


def evaluate_expression(expression: str) -> float:
    """Hitung ekspresi matematika dengan AST whitelist (tanpa ``eval``).

    Raises:
        ValueError: Jika ekspresi kosong, tidak valid, atau memakai sintaks
            yang tidak diizinkan.
    """
    cleaned = expression.strip().replace("^", "**")

    if not cleaned:
        raise ValueError("Expression is empty.")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc.msg}") from exc

    return _safe_eval(tree)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact numeric result.

    Supports + - * / // % ** parentheses and the functions abs, round, min,
    max, int, float, sqrt. Use this for any arithmetic; do not compute it
    yourself. Example: "(2 + 3) * 4 ** 2".
    """
    try:
        result = evaluate_expression(expression)
    except ValueError as exc:
        logger.warning("calculate rejected %r: %s", expression, exc)
        return f"Error: {exc}"

    logger.debug("calculate(%r) = %s", expression, result)
    return f"{expression} = {result}"


@tool
def current_datetime() -> str:
    """Return the current local date, time, weekday, and UTC time.

    Use this whenever the question depends on 'now', 'today', or the current
    year, because the model has no reliable knowledge of the present moment.
    """
    local_now = datetime.now()
    utc_now = datetime.now(timezone.utc)

    result = (
        f"Local: {local_now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()} "
        f"({local_now.strftime('%A')}), "
        f"UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"ISO week: {local_now.isocalendar().week}"
    )

    logger.debug("current_datetime() = %s", result)
    return result


def _is_within(root: Path, candidate: Path) -> bool:
    """Cegah path keluar dari proyek lewat ``..`` atau symlink."""
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_blocked(candidate: Path) -> bool:
    """True kalau file berisi kredensial atau bukan sumber kode proyek."""
    if candidate.name in BLOCKED_FILE_NAMES:
        return True

    if candidate.suffix in BLOCKED_SUFFIXES:
        return True

    return any(part in BLOCKED_DIRS for part in candidate.parts)


@tool
def read_project_file(relative_path: str) -> str:
    """Read a text file from this project and return its contents.

    The path is relative to the project root, for example "app.py" or
    "router.py". Use this to inspect the user's own code instead of guessing
    what a file contains. Credential files such as .env cannot be read.
    """
    candidate = (PROJECT_ROOT / relative_path).resolve()

    if not _is_within(PROJECT_ROOT, candidate):
        return "Error: path is outside the project directory."

    if _is_blocked(candidate):
        logger.warning("read_project_file blocked %r", relative_path)
        return "Error: reading this file is not allowed."

    if not candidate.is_file():
        return f"Error: file not found: {relative_path}"

    try:
        content = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error: cannot read {relative_path}: {exc}"

    if len(content) > MAX_FILE_CHARS:
        content = (
            content[:MAX_FILE_CHARS]
            + f"\n... [truncated, {len(content)} characters total]"
        )

    logger.debug("read_project_file(%r) -> %d chars", relative_path, len(content))
    return content


@tool
def list_project_files(subdirectory: str = ".") -> str:
    """List Python and config files in the project, or one of its folders.

    Use this before read_project_file when the file name is unknown. The
    argument is a folder relative to the project root; "." means the root.
    """
    candidate = (PROJECT_ROOT / subdirectory).resolve()

    if not _is_within(PROJECT_ROOT, candidate):
        return "Error: path is outside the project directory."

    if not candidate.is_dir():
        return f"Error: folder not found: {subdirectory}"

    skip_dirs = BLOCKED_DIRS | {".pytest_cache"}
    extensions = {".py", ".txt", ".toml", ".json", ".md"}

    found: list[str] = []

    for path in sorted(candidate.rglob("*")):
        if any(part in skip_dirs for part in path.parts):
            continue

        if (
            path.is_file()
            and path.suffix in extensions
            and not _is_blocked(path)
        ):
            try:
                relative = path.relative_to(PROJECT_ROOT)
            except ValueError:
                continue

            found.append(relative.as_posix())

            if len(found) >= MAX_LISTED_FILES:
                found.append("... [truncated]")
                break

    if not found:
        return f"No matching files under {subdirectory}"

    logger.debug("list_project_files(%r) -> %d entries", subdirectory, len(found))
    return "\n".join(found)


TOOLS = [calculate, current_datetime, read_project_file, list_project_files]
