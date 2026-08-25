"""Execute a notebook cell by cell, saving after every cell. Keeps Windows awake
for the duration (a per-process request that ends with the process).

Usage: python run_nb.py <in.ipynb> <out.ipynb>
"""

import ctypes
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
log = dst.with_suffix(".progress.log")
nb = nbformat.read(src, as_version=4)
t0 = time.perf_counter()


def note(msg):
    with log.open("a", encoding="utf-8") as f:
        f.write(f"[{time.perf_counter() - t0:9.1f}s] {msg}\n")


def on_cell_executed(cell=None, cell_index=None, execute_reply=None):
    nbformat.write(nb, dst)
    note(f"cell {cell_index} done ({cell.cell_type})")


log.write_text("", encoding="utf-8")
note(f"executing {src} -> {dst} ({len(nb.cells)} cells)")
client = NotebookClient(
    nb,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(src.parent)}},
    on_cell_executed=on_cell_executed,
)
try:
    client.execute()
except Exception as exc:
    nbformat.write(nb, dst)
    note(f"FAILED: {type(exc).__name__}: {exc}")
    print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise
nbformat.write(nb, dst)
note("complete")
print(f"complete in {(time.perf_counter() - t0) / 60:.1f} min -> {dst}")
