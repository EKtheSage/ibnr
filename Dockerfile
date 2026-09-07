# ibnr compute image: the package + cmdstan + every gallery Stan model
# COMPILED IN, so a container start pays zero toolchain cost. This is the
# image other services call today (scripts/) and the base the hosted scoring
# API will run on - both go through kernels/harness.py, which parallelizes
# across all visible cores by default.
#
#   docker build -t ibnr .
#   docker run --rm -e GH_TOKEN=<token with repo read> \
#     -e IBNR_MAX_WORKERS=8 --cpus 8 \
#     -v ibnr-cache:/data/ibnr-cache -v "$PWD/results:/app/analysis/results" \
#     ibnr python scripts/meyers_validation.py --model compartmental --per-line 50
#
# IBNR_MAX_WORKERS should match --cpus: inside a cpu-limited container,
# os.cpu_count() still reports the HOST's cores, so the env var is how the
# orchestrator tells the harness its real budget.

FROM python:3.12-slim

# build-essential: cmdstan is compiled into the image below (and gallery
# models compile against it). gh: the FALLBACK transport for gold-mart
# releases. The data repo is public and the adapter fetches over anonymous
# HTTPS, so gh is not needed for a normal run; it is kept because it is the
# only way past the GitHub API's unauthenticated rate limit (set GH_TOKEN),
# which a container fleet sharing one egress IP can realistically hit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
         -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
         > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The environment is installed by uv FROM uv.lock, never resolved by pip.
# `pip install ".[bayesian]"` no longer resolves at all (issue #132):
# bayesblend 0.0.8 pins arviz<0.19, arviz 0.18 caps numpy<2, and pip
# backtracks numpyro into ancient releases pinning jaxlib wheels that do not
# exist for cp312, then gives up. This repo steps over those stale caps with
# [tool.uv] override-dependencies - a uv concept pip cannot see - so the only
# faithful install is the locked resolution every CI leg already tests.
# --frozen installs exactly what uv.lock says and refuses a lockfile that has
# drifted from pyproject; --no-default-groups keeps dev/test tooling out of a
# compute image. The uv binary is copied from its release image, pinned like
# every other version here.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# Use the image's own interpreter (never a uv-managed download), and keep
# uv's cache out of the layer.
ENV UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# Dependency layer first: a src/ edit must not re-download the stack.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --extra bayesian --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-default-groups --extra bayesian --no-editable

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# cas-schedule-p is installed HERE and is deliberately not a dependency of the
# wheel. ibnr itself never imports it; only scripts/ does, to read the Meyers
# company selection rule out of `cas_schedule_p.screens`, and a reserving
# library should not pull a 17 MB wheel of one regulator's filings, about 20 MB
# of parquet once installed, into every install. But the image's own CMD runs
# one of those scripts, so without this line the default command of the image
# stops at ModuleNotFoundError.
# Pinned exactly, not floored, because the wheel carries the mart: the vintage
# decides which companies a run selects, so a floating version would change
# published results with no change to any code. tests/test_compute_image.py
# holds this pin equal to uv.lock's and re-checks that every script the image
# promises can still start on what these install lines put there.
# --no-deps so this one line cannot re-resolve anything the lock decided; the
# wheel is pure data with zero runtime dependencies, so there is nothing to
# resolve anyway.
RUN uv pip install --no-deps "cas-schedule-p==2026.6.13"

# cmdstan pinned to the version the reference results were produced with
# (CLAUDE.md: 2.39.0 on the dev machine)
RUN install_cmdstan --version 2.39.0 --cores 4

# compile every gallery Stan program into the image: containers start ready,
# and parallel workers never race the compiler (kernels.harness.precompile)
RUN python -c "from ibnr.kernels.harness import precompile; precompile()"

COPY scripts ./scripts
RUN mkdir -p analysis/results

# gold-mart release cache: mount a volume here so repeat runs skip the download
ENV IBNR_CACHE_DIR=/data/ibnr-cache

CMD ["python", "scripts/meyers_validation.py", "--help"]
