"""The compute image has to be able to run the scripts it advertises.

The Dockerfile builds the image other services call, and its own ``CMD`` is
``python scripts/meyers_validation.py --help``. That script imports
``cas_schedule_p.screens`` at module level, and the image installed
``.[bayesian]`` and nothing else, so the default command of the image as written
would stop at ``ModuleNotFoundError: No module named 'cas_schedule_p'`` before
printing a line of help. The image itself was never built, here or in CI (there
is no Docker on the development machine, which is why the Dockerfile has said
"build untested" since it landed); the failure was measured by running that
command with ``cas_schedule_p`` absent from the import path.

Nothing in this repository could have noticed. ``cas-schedule-p`` sits in the
``test`` dependency group, so every developer machine and every CI leg has it,
and the one environment that does not is the one nobody runs pytest in. The data
package is deliberately not a dependency of the wheel: ibnr itself never imports
it, only ``scripts/`` does, and a reserving library should not pull a 17 MB
wheel of one regulator's filings, about 20 MB of parquet once installed, into
every install. So it belongs in the image, pinned, and these tests are what keep
it there.

The checks reconstruct the image's install set from the Dockerfile itself rather
than from a hand-written list:

* the project and the extras it is installed with are read off the ``uv sync``
  lines, and every requirement on every ``pip install`` line is collected
  beside them. The distribution names all of those reach are computed from the
  metadata installed here, with markers evaluated once per requested extra;
* each script the image promises is then started as ``<script> --help`` in a
  subprocess whose ``sys.meta_path`` refuses any module whose distribution is
  installed on this machine but outside that set. The guarantee runs in one
  direction: everything the image would not have is refused, which is the
  failure that matters, but a package the image HAS and this leg does not
  cannot be supplied, and if a script ever imports one the failure message
  says which of the two cases it is;
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
import shlex
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
#: The first three are the scripts CLAUDE.md's 2026-08-04 paragraph names as
#: importing ``meyers_validation``, and each of them picks its companies with
#: the rule that module re-exports, so each of them needs the data package. A
#: fourth script joined that needs neither: ``benchmark_conventional.py`` has an
#: argparse command line and imports only numpy, pandas and ibnr, so the image
#: can start it although it never reads the Schedule P mart.
ALSO_PROMISED = (
    "compare_gallery.py",
    "heldout_leaderboard.py",
    "parity_gallery.py",
    "benchmark_conventional.py",
)

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
    "conventional_examples.py": (
        "a loader with no command line, imported by the conventional benchmark "
        "driver rather than run on its own. Its only runtime input is a network "
        "download of the published appendix, which the image cannot make."
    ),
    "conventional_synthetic.py": (
        "a portfolio generator with no command line, imported by the "
        "conventional benchmark driver rather than run on its own."
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


#: Where one command on a ``RUN`` line ends and the next begins. A parser that
#: walked past these would read the second command's arguments as the first's.
_SEPARATORS = {"&&", "||", "|", ";"}

#: Options that would make a ``uv sync`` install more than the extras this file
#: reads off it. Any of them means the parser below has stopped describing the
#: image, so it refuses rather than quietly under-reporting.
_WIDENING_SYNC_OPTIONS = (
    "--all-extras",
    "--all-groups",
    "--group",
    "--only-group",
    "--dev",
    "--all-packages",
)


def _command_lines() -> list[str]:
    """The Dockerfile's commands, one per line, ready to be tokenized.

    Continuation lines are joined first, so a command wrapped across several
    lines of a ``RUN`` is read whole, and comment lines are dropped so that
    prose in the header cannot be mistaken for an install.
    """
    text = "\n".join(
        line
        for line in _dockerfile().replace("\r\n", "\n").splitlines()
        if not line.lstrip().startswith("#")
    ).replace("\\\n", " ")
    return text.splitlines()


def _install_specs() -> list[str]:
    """Every requirement on every ``pip install`` line.

    Reading ALL of them is the point. A parser that stops at the first quoted
    string on a line reads ``uv pip install --no-deps "cas-schedule-p==..."``
    as installing the flag, or stops before a second requirement written beside
    it, either of which makes a present fix look absent.

    The line is tokenized the way a shell would, so a requirement counts whether
    it is written in double quotes, single quotes or bare, and tokens starting
    with ``-`` are dropped as pip's own flags. ``uv pip install`` is read by the
    same parser as a plain ``pip install``, because the tokens ``pip install``
    appear in both.
    """
    specs: list[str] = []
    for line in _command_lines():
        if "pip install" not in line:
            continue
        tokens = shlex.split(line)
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] != "pip" or tokens[i + 1] != "install":
                i += 1
                continue
            i += 2
            while i < len(tokens) and tokens[i] not in _SEPARATORS:
                if not tokens[i].startswith("-"):
                    specs.append(tokens[i])
                i += 1
    return specs


def _sync_extras() -> tuple[str, ...]:
    """The extras the Dockerfile's ``uv sync`` lines install the project with.

    The image is built from the lockfile rather than resolved by pip (issue
    #132), so the project and its dependencies arrive through ``uv sync`` and
    the extras named there are what decide the install set. Each sync line is
    checked as it is read, because every one of these options changes what the
    image holds while leaving the extras looking the same.

    ``uv sync`` is declarative: it makes the environment match what the line
    asks for, so a later line naming fewer extras REMOVES what an earlier one
    installed. Two lines that disagree are therefore an image bug rather than a
    parser nuisance, and the identical set is required of all of them.
    """
    requested: list[tuple[str, ...]] = []
    for line in _command_lines():
        tokens = shlex.split(line)
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] != "uv" or tokens[i + 1] != "sync":
                i += 1
                continue
            i += 2
            extras: list[str] = []
            options: list[str] = []
            while i < len(tokens) and tokens[i] not in _SEPARATORS:
                option, _, attached = tokens[i].partition("=")
                options.append(option)
                if option == "--extra":
                    if attached:
                        extras.append(attached)
                    else:
                        i += 1
                        assert i < len(tokens), f"--extra names no extra on {line!r}"
                        extras.append(tokens[i])
                i += 1
            assert "--frozen" in options, (
                f"the uv sync on {line!r} is not --frozen, so it re-resolves the "
                "dependencies instead of installing the locked resolution. That "
                "lock is the only faithful install of this project (issue #132): "
                "it carries the [tool.uv] override-dependencies that step over "
                "arviz's and bermuda's stale numpy caps, and it is exactly what "
                "every CI leg tests."
            )
            assert "--no-default-groups" in options, (
                f"the uv sync on {line!r} does not pass --no-default-groups, so "
                "it installs the default dev group too. The image does not carry "
                "that tooling, so the install set computed here would be smaller "
                "than the image and the refusal below would let packages through "
                "that the image really lacks."
            )
            widening = [option for option in _WIDENING_SYNC_OPTIONS if option in options]
            assert not widening, (
                f"the uv sync on {line!r} passes {widening}, which install more "
                "than the --extra values this parser reads. Extend the parser to "
                "understand them before widening the image, or the install set "
                "here silently stops matching what the Dockerfile builds."
            )
            requested.append(tuple(sorted(extras)))
    assert requested, (
        "the Dockerfile has no uv sync line at all. The image installs the "
        "project and its locked dependencies through uv sync, so if that "
        "changed, rewrite this parser to read whatever replaced it."
    )
    assert len(set(requested)) == 1, (
        "the Dockerfile's uv sync lines ask for different extras: "
        f"{sorted(set(requested))}. uv sync makes the environment match the line "
        "it is given, so the last one wins and every extra the earlier lines "
        "installed and it omits is REMOVED again."
    )
    return requested[0]


def _cmd_argv() -> list[str]:
    """The image's default command, read as the JSON array it is written as."""
    matches = re.findall(r"^CMD\s+(\[.*\])\s*$", _dockerfile(), flags=re.MULTILINE)
    assert len(matches) == 1, f"expected exactly one JSON-form CMD line, found {matches}"
    return json.loads(matches[0])


def _reachable_distributions(name: str, extras: tuple[str, ...]) -> set[str]:
    """Distribution names reachable from ``name[extras]`` through local metadata.

    A requirement whose distribution is not installed on this machine still
    counts by name; only its own dependencies are unknown, because there is no
    metadata here to read them from. The direct requirements of every extra the
    Dockerfile asks for are therefore always present, since they come from
    ibnr's own metadata, which is installed wherever this file runs. What can be
    missing is the layer below: on a leg that does not sync ``[bayesian]``, the
    packages arviz and numpyro bring with them (matplotlib, xarray, tqdm and the
    rest) are not reachable, so the set is smaller there than in the image. That
    costs nothing today, because no script the image promises imports one of
    them, and the checks below say so when it starts to matter rather than
    blaming the Dockerfile for it.
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
    """Everything the Dockerfile's uv sync and pip install lines put in the image."""
    names = _reachable_distributions("ibnr", _sync_extras())
    for spec in _install_specs():
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


def _distribution_names(module: str, modules: dict[str, list[str]]) -> list[str]:
    """Which distributions a top-level module belongs to, by its own name if none.

    ``packages_distributions()`` only knows what is installed here, and the fall
    back to the module's own name is what keeps a distribution the image has and
    this leg does not, cmdstanpy on the core leg for instance, from reading as a
    package the image lacks.
    """
    return [canonicalize_name(d) for d in modules.get(module, [])] or [canonicalize_name(module)]


def test_the_refusal_can_actually_refuse_something():
    """Guard the guard.

    The subprocess check below refuses a module by looking its top-level name up
    in ``packages_distributions()`` and asking whether that distribution is in
    the image's install set. Three ways it could pass while testing nothing: the
    map comes back without the names the check leans on, the install set turns
    out to hold everything this machine has, or the extras are read wrongly and
    the set is not the image's. The first two are checked on pytest, which every
    CI leg installs and the image must never have. The third is checked on the
    extras themselves: the Dockerfile asks for ``[bayesian]`` and for no other,
    so a requirement carried by that extra has to be in and a requirement
    carried by ``[nn]`` or ``[interop]`` has to be out. Those three come from
    ibnr's own metadata, which is installed wherever this file runs, so this
    reads the same on every leg.
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
        "that picks its companies fails at import inside the image"
    )
    assert "cmdstanpy" in image, (
        "cmdstanpy is a requirement of ibnr[bayesian] and the Dockerfile asks "
        "for that extra, so it has to be in the install set. It is not, which "
        "means the extras on the uv sync lines are not being read and the "
        "set is smaller than the image"
    )
    for extra, distribution in (("nn", "torch"), ("interop", "chainladder")):
        assert distribution not in image, (
            f"{distribution} is a requirement of ibnr[{extra}], an extra the "
            "Dockerfile does not ask for, yet it is in the install set. That "
            "means every extra is being treated as requested and the set is "
            "bigger than the image"
        )
    assert any(d in image for d in _distribution_names("cmdstanpy", modules)), (
        "cmdstanpy is in the image's install set and still grades as outside "
        "it. On a leg that does not sync [bayesian] there is no entry for it "
        "in packages_distributions(), so without the fall back to the module's "
        "own name the static scan below would report the image as missing a "
        "package the Dockerfile installs"
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
                f"{dists[0]!r} is outside what the Dockerfile's install "
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


def _run_in_image(script: Path) -> subprocess.CompletedProcess[str]:
    """Run ``<script> --help`` in the refusing child. One code path, two callers.

    The positive control below and the promised-script runs go through this
    function together, so the control cannot pass on a child the real runs never
    use.
    """
    return subprocess.run(
        [sys.executable, "-c", _SMOKE, json.dumps(sorted(_image_distributions())), str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )


#: A plain import failure, as opposed to the child finder's own refusal, whose
#: wording is different on purpose.
_PLAIN_IMPORT_FAILURE = re.compile(r"No module named '([^']+)'")


def _why_it_failed(stderr: str, image: frozenset[str]) -> str:
    """Name which of the two failures this is, so the remedy is the right one.

    The child refuses what the image lacks, and that refusal says so in its own
    words. The other way an import can fail there is a package the image HAS and
    this test environment does not, which the child cannot conjure up: a leg
    that syncs no extras has none of ``[bayesian]``. Sending someone to add such
    a package to the Dockerfile, which already installs it, is the wrong
    direction entirely, so it is named here instead.
    """
    names = _PLAIN_IMPORT_FAILURE.findall(stderr)
    if not names:
        return ""
    top = names[-1].split(".")[0]
    if canonicalize_name(top) in image:
        return (
            f"\n{top!r} IS in the image's install set. What is missing is this "
            "test environment's copy of it, which is a fact about the CI leg "
            "and not about the Dockerfile: run the leg that installs the extra "
            "carrying it. Do NOT add it to the image, which has it already."
        )
    return ""


def test_a_failed_run_says_which_of_the_two_failures_it_is():
    """The two ways the child can fail need opposite remedies.

    Checked on the strings rather than by arranging each failure, because
    arranging the second one means a package that is absent here and present on
    another leg, which would make the test read differently depending on where
    it runs. cmdstanpy is the example either way: it is a requirement of
    ibnr[bayesian], so it is in the image on every leg, and it is installed only
    on the legs that sync that extra.
    """
    image = _image_distributions()
    refused = (
        "ModuleNotFoundError: 'pytest' is not in the compute image: its "
        "distribution 'pytest' is outside what the Dockerfile's install "
        "lines put there"
    )
    assert _why_it_failed(refused, image) == "", (
        "the child's own refusal is already the right message and must not be "
        "second-guessed: that module really is outside the image"
    )
    assert _why_it_failed("ModuleNotFoundError: No module named 'torch'", image) == "", (
        "torch is outside the image too, so a plain failure on it needs no further explanation"
    )
    leg = _why_it_failed("ModuleNotFoundError: No module named 'cmdstanpy'", image)
    assert "IS in the image's install set" in leg and "Do NOT add it to the image" in leg, (
        "a package the image installs went missing because this leg does not "
        "have it, and the message does not say so. Read literally, the run "
        "above then tells someone to add cmdstanpy to a Dockerfile that "
        f"installs it already. Got: {leg!r}"
    )


def test_the_refusing_child_refuses(tmp_path):
    """The child is the mechanism, so run it once on something it must refuse.

    Without this the four runs below prove nothing on their own: with the
    Dockerfile fixed, a child whose refusal never fires passes every one of
    them, because the scripts then import only packages the image really has.
    pytest is the probe because it is installed wherever this file runs and is
    outside the image on purpose.
    """
    script = tmp_path / "imports_something_the_image_lacks.py"
    script.write_text("import pytest\n", encoding="utf-8")
    proc = _run_in_image(script)
    assert proc.returncode != 0, (
        "the child imported pytest, which the image does not install, and "
        f"exited cleanly. It is refusing nothing:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "'pytest' is not in the compute image" in proc.stderr, (
        "the child failed, but not with its own refusal, so the run below "
        f"would not be testing the image's install set:\n{proc.stderr}"
    )


@pytest.mark.parametrize("name", _promised_scripts())
def test_every_promised_script_starts_on_what_the_image_installs(name):
    """``<script> --help`` must reach argparse with only the image's packages.

    ``--help`` is the cheapest command that still executes every module-level
    import, which is where this failure lives: the image's CMD is itself a
    ``--help``, and it was the ``--help`` that stopped.
    """
    script = SCRIPTS / name
    assert script.exists(), f"scripts/{name} is promised by this test but does not exist"
    proc = _run_in_image(script)
    assert proc.returncode == 0, (
        f"scripts/{name} --help cannot start inside the compute image:\n"
        f"{proc.stdout}\n{proc.stderr}"
        f"{_why_it_failed(proc.stderr, _image_distributions())}"
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
    own name as the distribution name: that is what keeps a distribution which
    is in the image but absent from this leg, such as cmdstanpy on the core leg,
    from being reported as a package the image lacks.
    """
    image = _image_distributions()
    modules = md.packages_distributions()
    outside = set()
    for name in _promised_scripts():
        for where, top in _third_party_imports(name):
            if not any(dist in image for dist in _distribution_names(top, modules)):
                outside.add((where, top))
    assert outside == set(ALLOWED_OUTSIDE_IMAGE), (
        "the set of imports the compute image cannot satisfy changed.\n"
        f"  newly outside the image: {sorted(outside - set(ALLOWED_OUTSIDE_IMAGE))}\n"
        f"  no longer outside:       {sorted(set(ALLOWED_OUTSIDE_IMAGE) - outside)}\n"
        "Either install the package in the image, move the import into the "
        "branch that needs it, or add it above with the reason it is accepted. "
        "One case first, before doing any of those: a package that reaches the "
        "image underneath arviz, numpyro or pymc is only reachable here on a "
        "leg that syncs [bayesian] (see _reachable_distributions), so check the "
        "Dockerfile before concluding the image lacks it."
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
