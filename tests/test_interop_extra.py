"""The optional interop dependencies must fail legibly, not with a bare import error.

``chainladder`` and ``bermuda`` are not core dependencies. On a plain
``pip install ibnr`` they are absent, and before the ``interop`` extra existed
``to_chainladder()``/``to_bermuda()`` died with a naked
``ModuleNotFoundError: No module named 'chainladder'`` that said nothing about
how to fix it.

The dev environment *has* both installed, so the missing case is simulated by
patching ``importlib.import_module`` (which is what ``_require_interop`` calls;
patching ``builtins.__import__`` would not intercept it).
"""

import importlib

import pytest

from ibnr.triangle.io import _require_interop


def test_require_interop_returns_the_module_when_present():
    assert _require_interop("json", "feature").__name__ == "json"


@pytest.mark.parametrize(
    ("module", "feature"),
    [("chainladder", "Triangle.to_chainladder()"), ("bermuda", "Triangle.to_bermuda()")],
)
def test_require_interop_names_the_feature_and_the_extra(monkeypatch, module, feature):
    def missing(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(ModuleNotFoundError) as exc:
        _require_interop(module, feature)

    msg = str(exc.value)
    assert feature in msg  # which call failed
    assert module in msg  # what is missing
    assert "ibnr[interop]" in msg  # how to fix it
    assert exc.value.__cause__ is not None  # original error preserved
