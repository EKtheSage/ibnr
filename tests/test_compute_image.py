"""The compute image has to be able to run the scripts it advertises.

The Dockerfile builds the image other services call, and its own ``CMD`` is
``python scripts/meyers_validation.py --help``. That script imports
``cas_schedule_p.screens`` at module level, and the image installed
``.[bayesian]`` and nothing else, so the published image's default command died
with ``ModuleNotFoundError: No module named 'cas_schedule_p'`` before printing a
line of help.

Nothing in this repository could have noticed. ``cas-schedule-p`` sits in the
``test`` dependency group, so every developer machine and every CI leg has it,
and the one environment that does not is the one nobody runs pytest in. The data
package is deliberately not a dependency of the wheel: ibnr itself never imports
it, only ``scripts/`` does, and a reserving library should not pull a 17 MB
parquet of one regulator's filings into every install. So it belongs in the
image, pinned, and these tests are what keep it there.

The checks reconstruct the image's install set from the Dockerfile itself rather
than from a hand-written list:

* every quoted requirement on every ``pip install`` line is collected, and the
  distribution names those requirements reach are computed from the metadata
  installed here, with markers evaluated once per requested extra;
* each script the image promises is then started as ``<script> --help`` in a
  subprocess whose ``sys.meta_path`` refuses any module whose distribution is
  installed on this machine but outside that set. That is a stand-in for the
  image, and an exact one for the failure that matters: an import of something
  the image does not carry;
* a static scan then reads every import in those scripts at any nesting depth,
  following imports of sibling scripts, because a package needed only by a
  branch that ``--help`` never reaches is still a package the image needs.

Nothing here can skip. The refusing subprocess needs no optional extra
installed; it needs the opposite, and the more a leg installs the more the
refusal has to say no to.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from importlib import metadata as md
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "Dockerfile"
SCRIPTS = REPO / "scripts"

#: The Meyers company selection rule, its SQL and the gold mart are one release
#: artifact with one version number (CLAUDE.md, 2026-08-04). Every script that
#: selects companies reads it from here.
DATA_PACKAGE = "cas-schedule-p"

#: Scripts besides the CMD's own that the image is expected to be able to run.
#: These are the three the harness documentation and the analysis notebooks call
#: for a retrospective, and each of them selects companies, so each of them
#: needs the data package.
ALSO_PROMISED = ("compare_gallery.py", "heldout_leaderboard.py", "parity_gallery.py")

#: Scripts the image is NOT expected to run, each with the reason. Named rather
#: than inferred, so a new script cannot join scripts/ without someone deciding
#: which side of this line it falls on.
NOT_PROMISED = {
    "benchmark_speed.py": (
        "imports chainladder at module level, because it times ibnr against "
        "chainladder-python. That needs the [interop] extra, which the image "
        "does not install and is not meant to: the image runs studies, it does "
        "not benchmark against another library."
    ),
    "benchmark_scaling.py": (
        "imports benchmark_speed, so it inherits that script's chainladder "
        "requirement for the same reason."
    ),
    "lint_md_snippets.py": (
        "a lint tool for this repository's own markdown, run by the lint CI "
        "job. It needs a ruff binary, which the image does not copy, and it "
        "computes nothing."
    ),
}

#: (script file, top-level module) pairs the static scan may find outside the
#: image's install set, each with the reason it is accepted. Compared as a SET,
#: so an entry that stops being found is as much a finding as a new one: it
#: means this list has gone stale and stopped describing the scripts.
ALLOWED_OUTSIDE_IMAGE = {
    ("compare_gallery.py", "torch"): (
        "a nested import inside the exposure-power diagnostic, reached only "
        "with --nn-exposure-sigma. The image installs [bayesian] and not [nn], "
        "so the neural arms of that comparison do not run there; the Bayesian "
        "arms, which are what the image exists for, do."
    ),
}


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _install_specs() -> list[str]:
    """Every quoted requirement on every ``pip install`` line.

    Reading ALL of them is the point. A parser that stops at the first quoted
    string on a line reads ``pip install ".[bayesian]" "cas-schedule-p==..."``
    as installing only the first, which makes a present fix look absent.
    Comment lines are dropped so that prose in the header cannot be mistaken for
    an install.
    """
    return [
        spec
        for line in _dockerfile().splitlines()
        if "pip install" in line and not line.lstrip().startswith("#")
        for spec in re.findall(r'"([^"]+)"', line)
    ]


def _cmd_argv() -> list[str]:
    """The image's default command, read as the JSON array it is written as."""
    matches = re.findall(r"^CMD\s+(\[.*\])\s*$", _dockerfile(), flags=re.MULTILINE)
    assert len(matches) == 1, f"expected exactly one JSON-form CMD line, found {matches}"
    return json.loads(matches[0])


def _reachable_distributions(name: str, extras: tuple[str, ...]) -> set[str]:
    """Distribution names reachable from ``name[extras]`` through local metadata.

    A requirement whose distribution is not installed on this machine still
    counts by name; only its own dependencies are unknown. That is the right
    reading: the image installs from PyPI and gets those dependencies whether or
    not this machine has them, and a stricter rule would make the answer depend
    on which extras the CI leg happened to sync.
    """
    seen: set[tuple[str, frozenset[str]]] = set()
    stack = [(canonicalize_name(name), frozenset(extras))]
    while stack:
        dist, requested = stack.pop()
        if (dist, requested) in seen:
            continue
        seen.add((dist, requested))
        try:
            requires = md.requires(dist) or []
        except md.PackageNotFoundError:
            continue
        for raw in requires:
            req = Requirement(raw)
            wanted = requested or frozenset({""})
            if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in wanted):
                stack.append((canonicalize_name(req.name), frozenset(req.extras)))
    return {dist for dist, _ in seen}


def _image_distributions() -> frozenset[str]:
    """Everything the Dockerfile's pip install lines put in the image."""
    specs = _install_specs()
    assert specs, "the Dockerfile has no pip install line at all"
    names: set[str] = set()
    for spec in specs:
        if spec.startswith("."):
            # the project itself, built from the copied source tree
            extras = re.findall(r"\[([^\]]*)\]", spec)
            requested = tuple(e.strip() for e in extras[0].split(",")) if extras else ()
            names |= _reachable_distributions("ibnr", requested)
        else:
            req = Requirement(spec)
            names |= _reachable_distributions(req.name, tuple(req.extras))
    return frozenset(names)


def _promised_scripts() -> tuple[str, ...]:
    """The CMD's own script first, then the three named above."""
    from_cmd = [arg for arg in _cmd_argv() if arg.endswith(".py")]
    assert len(from_cmd) == 1, f"expected exactly one script in CMD, found {from_cmd}"
    return tuple(dict.fromkeys([Path(from_cmd[0]).name, *ALSO_PROMISED]))


def _spec_for(distribution: str) -> Requirement | None:
    for spec in _install_specs():
        if spec.startswith("."):
            continue
        req = Requirement(spec)
        if canonicalize_name(req.name) == canonicalize_name(distribution):
            return req
    return None


def _locked_version(distribution: str) -> str:
    with open(REPO / "uv.lock", "rb") as fh:
        lock = tomllib.load(fh)
    (pkg,) = [
        p
        for p in lock["package"]
        if canonicalize_name(p["name"]) == canonicalize_name(distribution)
    ]
    return pkg["version"]


def test_the_image_pins_the_data_package_to_the_locked_version():
    """The image must install the data package, at exactly the locked version.

    An exact pin rather than a floor, because the wheel carries the mart: which
    companies a published run selected is decided by that vintage, so a floating
    version changes results without changing a line of code.
    """
    req = _spec_for(DATA_PACKAGE)
    assert req is not None, (
        f"the Dockerfile installs {_install_specs()} and none of them is "
        f"{DATA_PACKAGE}. scripts/meyers_validation.py imports "
        "cas_schedule_p.screens at module level and the image's own CMD runs "
        "that script, so the default command of the built image cannot start."
    )
    pins = list(req.specifier)
    assert len(pins) == 1 and pins[0].operator == "==", (
        f"expected an exact pin for {DATA_PACKAGE}, got {str(req)!r}"
    )
    assert pins[0].version == _locked_version(DATA_PACKAGE), (
        f"the image pins {pins[0].version} and uv.lock resolves "
        f"{_locked_version(DATA_PACKAGE)}, so a run inside the image would read "
        "a different mart vintage from a run on a developer machine."
    )


def test_the_refusal_can_actually_refuse_something():
    """Guard the guard.

    The subprocess check below refuses a module by looking its top-level name up
    in ``packages_distributions()`` and asking whether that distribution is in
    the image's install set. Two ways it could pass while testing nothing: the
    map comes back without the names the check leans on, or the install set
    turns out to hold everything this machine has. Both are checked here
    directly, on a package that is installed in every CI leg and must never be
    in the image.
    """
    modules = md.packages_distributions()
    assert [canonicalize_name(d) for d in modules.get("pytest", [])] == ["pytest"], (
        "pytest does not map to a distribution here, so the refusing "
        "subprocess would refuse nothing at all"
    )
    assert [canonicalize_name(d) for d in modules.get("cas_schedule_p", [])] == [DATA_PACKAGE], (
        "cas_schedule_p does not map to its distribution here, so the check "
        "could not tell whether the image carries it"
    )
    image = _image_distributions()
    assert "pytest" not in image, (
        "pytest is inside the image's install set, which means the set is not "
        "the image's and the refusal would let this whole machine through"
    )
    assert DATA_PACKAGE in image, (
        f"{DATA_PACKAGE} is outside the image's install set, so every script "
        "that selects companies fails at import inside the image"
    )


#: Child program for the smoke test. Takes the image's install set as JSON and
#: the script path on argv, installs a meta-path finder that refuses anything
#: outside that set, then runs the script's ``--help`` exactly as the image's
#: CMD would. Written as a subprocess because this process has already imported
#: half the packages the finder is supposed to refuse.
_SMOKE = """
import json, runpy, sys
from importlib import metadata as md
from packaging.utils import canonicalize_name

image = set(json.loads(sys.argv[1]))
script = sys.argv[2]
dist_of = {
    module: [canonicalize_name(d) for d in dists]
    for module, dists in md.packages_distributions().items()
}


class _NotInTheImage:
    "Refuses what the compute image would not have installed."

    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        dists = dist_of.get(top)
        if dists and not any(d in image for d in dists):
            raise ModuleNotFoundError(
                f"{top!r} is not in the compute image: its distribution "
                f"{dists[0]!r} is outside what the Dockerfile's pip install "
                f"lines put there",
                name=name,
            )
        return None


sys.meta_path.insert(0, _NotInTheImage())
sys.argv = [script, "--help"]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    sys.exit(exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1))
"""


@pytest.mark.parametrize("name", _promised_scripts())
def test_every_promised_script_starts_on_what_the_image_installs(name):
    """``<script> --help`` must reach argparse with only the image's packages.

    ``--help`` is the cheapest command that still executes every module-level
    import, which is where this failure lives: the image's CMD is itself a
    ``--help``, and it was the ``--help`` that crashed.
    """
    script = SCRIPTS / name
    assert script.exists(), f"scripts/{name} is promised by this test but does not exist"
    proc = subprocess.run(
        [sys.executable, "-c", _SMOKE, json.dumps(sorted(_image_distributions())), str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"scripts/{name} --help cannot start inside the compute image:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def _third_party_imports(name: str) -> set[tuple[str, str]]:
    """(file, top-level module) for every import in a script and its siblings.

    ``ast.walk`` reaches every nesting depth, so an import written inside a
    function to keep the module light is still found. An import of a sibling
    script is followed rather than reported, because running the script runs it
    too: ``parity_gallery.py`` reaches ``cas_schedule_p`` only that way, through
    an import of ``meyers_validation`` written inside a function.
    """
    stdlib = set(sys.stdlib_module_names)
    siblings = {p.stem: p.name for p in SCRIPTS.glob("*.py")}
    found: set[tuple[str, str]] = set()
    pending, visited = [name], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for node in ast.walk(ast.parse((SCRIPTS / current).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                tops = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops = [node.module.split(".")[0]]
            else:
                continue
            for top in tops:
                if top in stdlib or top == "ibnr":
                    continue
                if top in siblings:
                    pending.append(siblings[top])
                else:
                    found.add((current, top))
    return found


def test_the_import_scan_reaches_nested_and_sibling_imports():
    """How far the scan reaches, pinned on the two cases that need it.

    ``parity_gallery.py`` names no data package of its own. It imports
    ``meyers_validation`` inside a function, and that module imports
    ``cas_schedule_p`` at its top. A scan that read only module-level imports,
    or only the file it was handed, would report that script as needing nothing
    and be wrong about the one package this file exists to watch.
    """
    assert ("meyers_validation.py", "cas_schedule_p") in _third_party_imports("parity_gallery.py")
    assert ("compare_gallery.py", "torch") in _third_party_imports("heldout_leaderboard.py")


def test_no_promised_script_needs_a_package_the_image_lacks():
    """Every import in the promised scripts, at any depth, against the image.

    The subprocess check above only sees what ``--help`` executes. This one
    reads the source, so a package a real run needs on a branch ``--help`` never
    takes is still found. A module this machine does not have falls back to its
    own name as the distribution name, which is how ``torch`` is graded in a leg
    that never installs it.
    """
    image = _image_distributions()
    modules = md.packages_distributions()
    outside = set()
    for name in _promised_scripts():
        for where, top in _third_party_imports(name):
            dists = [canonicalize_name(d) for d in modules.get(top, [])] or [canonicalize_name(top)]
            if not any(dist in image for dist in dists):
                outside.add((where, top))
    assert outside == set(ALLOWED_OUTSIDE_IMAGE), (
        "the set of imports the compute image cannot satisfy changed.\n"
        f"  newly outside the image: {sorted(outside - set(ALLOWED_OUTSIDE_IMAGE))}\n"
        f"  no longer outside:       {sorted(set(ALLOWED_OUTSIDE_IMAGE) - outside)}\n"
        "Either install the package in the image, move the import into the "
        "branch that needs it, or add it above with the reason it is accepted."
    )


def test_every_script_is_either_promised_or_excluded_with_a_reason():
    """No script may join scripts/ without a decision about the image.

    Without this, a new study script would be covered by nothing and the two
    checks above would keep passing while the image quietly stopped being able
    to run the newest thing anyone wrote.
    """
    on_disk = {p.name for p in SCRIPTS.glob("*.py")}
    accounted = set(_promised_scripts()) | set(NOT_PROMISED)
    assert on_disk == accounted, (
        "scripts/ and this file disagree.\n"
        f"  present but unaccounted for: {sorted(on_disk - accounted)}\n"
        f"  listed here but absent:      {sorted(accounted - on_disk)}"
    )
