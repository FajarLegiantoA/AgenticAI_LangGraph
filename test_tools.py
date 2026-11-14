"""Test untuk tool yang dipakai model.

Tool ini dieksekusi atas nama model, jadi pengujian fokus pada hal yang
paling berbahaya: ekspresi arbitrer, pembacaan file rahasia, dan path
traversal.
"""

from __future__ import annotations

import pytest

from tools import (
    TOOLS,
    evaluate_expression,
    list_project_files,
    read_project_file,
)

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2 ** 10", 1024),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("-5 + 2", -3),
        ("round(2.567, 2)", 2.57),
        ("abs(-7)", 7),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
    ],
)
def test_calculator_accepts_safe_arithmetic(expression, expected):
    assert evaluate_expression(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('secrets.txt').read()",
        "().__class__.__bases__",
        "eval('1+1')",
        "exec('x=1')",
        "[x for x in (1,2)]",
        "x + 1",
        "1 if 2 else 3",
        "'abc'.upper()",
        "",
        "2 +",
    ],
)
def test_calculator_rejects_unsafe_input(expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression)


def test_calculator_rejects_huge_exponent():
    # 9 ** 9 ** 9 akan menghabiskan memori tanpa batas.
    with pytest.raises(ValueError):
        evaluate_expression("9 ** 9 ** 9")


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def test_read_project_file_reads_real_file():
    content = read_project_file.invoke({"relative_path": "tools.py"})

    assert "evaluate_expression" in content


def test_read_project_file_blocks_env_file():
    result = read_project_file.invoke({"relative_path": ".env"})

    assert "not allowed" in result.lower()


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "secrets/.env",
        "credentials.json",
        "server.key",
        ".git/config",
    ],
)
def test_read_project_file_blocks_sensitive_paths(path):
    result = read_project_file.invoke({"relative_path": path})

    assert "not allowed" in result.lower()


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "..\\..\\Windows\\win.ini",
        "C:\\Windows\\win.ini",
    ],
)
def test_read_project_file_blocks_path_traversal(path):
    result = read_project_file.invoke({"relative_path": path})

    assert "outside" in result.lower() or "not allowed" in result.lower()


def test_list_project_files_returns_names():
    listing = list_project_files.invoke({"subdirectory": "."})

    assert "tools.py" in listing
    assert ".venv" not in listing


def test_list_project_files_blocks_traversal():
    result = list_project_files.invoke({"subdirectory": "../"})

    assert "outside" in result.lower() or "not allowed" in result.lower()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_tools_are_named_and_described():
    names = {tool.name for tool in TOOLS}

    assert names == {
        "calculate",
        "current_datetime",
        "read_project_file",
        "list_project_files",
    }

    for tool in TOOLS:
        assert tool.description.strip()
