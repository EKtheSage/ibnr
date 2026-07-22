# ibnr compute image: the package + cmdstan + every gallery Stan model
# COMPILED IN, so a container start pays zero toolchain cost. This is the
# image other services call today (scripts/) and the base the hosted scoring
# API will run on — both go through kernels/harness.py, which parallelizes
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
# models compile against it). gh: fetches gold-mart releases at runtime from
# the private data repo (auth via GH_TOKEN).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
         -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
         > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[bayesian]"

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
