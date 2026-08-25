"""Assemble the reworked notebook (and a smoke variant) from cells_v2.py."""

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cells_v2 import CELLS  # noqa: E402

SMOKE = "--smoke" in sys.argv
SMOKE_SUBS = [
    (
        "NN_CONFIG = dict(max_epochs=120, patience=15, ensemble_size=5, n_draws=500)",
        "NN_CONFIG = dict(max_epochs=2, patience=2, ensemble_size=2, n_draws=8)",
    ),
    (
        "ML_CONFIG = dict(max_epochs=120, patience=15, ensemble_size=5, n_draws=500)",
        "ML_CONFIG = dict(max_epochs=2, patience=2, ensemble_size=2, n_draws=8)",
    ),
    ("heldout_n_draws=10_000", "heldout_n_draws=200"),
]

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.12.13"}

n_sub = 0
for kind, src in CELLS:
    src = src.rstrip("\n")
    if SMOKE and kind == "code":
        for old, new in SMOKE_SUBS:
            if old in src:
                src = src.replace(old, new)
                n_sub += 1
    cell = (
        nbformat.v4.new_markdown_cell(src)
        if kind == "markdown"
        else nbformat.v4.new_code_cell(src)
    )
    nb.cells.append(cell)

out = HERE / "smoke.ipynb" if SMOKE else HERE.parent / "03c_multiline_transformer.ipynb"
nbformat.write(nb, out)
kinds = [k for k, _ in CELLS]
print(
    f"wrote {out} ({len(nb.cells)} cells: {kinds.count('markdown')} markdown, "
    f"{kinds.count('code')} code{f'; {n_sub} smoke substitutions' if SMOKE else ''})"
)
if SMOKE and n_sub != 3:
    raise SystemExit(f"expected 3 smoke substitutions, made {n_sub}")
