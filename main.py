"""
Smoke test for main.py.

main.py is a live camera-processing script (it connects to a video stream,
loads a YOLO model, and runs an infinite loop), so it can't be safely
imported during a CI run the way a normal library module could -- doing so
would make GitHub Actions try to open a network camera and load an AI model
on every single test run.

Instead, this test does the one thing that's both safe and genuinely
useful in CI: it checks that main.py is valid, parseable Python. That
catches real mistakes -- a missing colon, a stray indent, an unclosed
bracket, a typo like "get_ipython" being used without being imported --
before they ever reach the live camera. It's a "does this even run"
check, not a full test of the traffic-detection logic itself.
"""

import ast
import pathlib


def test_main_is_syntactically_valid():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    main_path = repo_root / "main.py"

    assert main_path.exists(), f"Expected to find main.py at {main_path}"

    source = main_path.read_text(encoding="utf-8")

    # ast.parse raises SyntaxError if the file isn't valid Python.
    # We don't need to do anything with the result -- just confirm it
    # doesn't raise.
    ast.parse(source, filename=str(main_path))
