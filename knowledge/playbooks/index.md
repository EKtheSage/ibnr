# Playbooks

* [Cut a release](cut-a-release.md) - The pre-flight checks run before tagging an ibnr release, the traps hit while publishing to PyPI, and the release history from 0.2.0 to 0.7.1 as the agent's status note recorded it.
* [Making an analysis notebook portable](make-a-notebook-portable.md) - How to make an analysis notebook run from any directory on a plain pip install: vendor repo scripts, pin the data publish explicitly (a bare pinned_source() does call gh), assert the package version, and prove it from a neutral directory and a fresh PyPI venv.
* [Re-run notebook 04](rerun-notebook-04.md) - How long each stage of analysis/04 takes, and how to run it in pieces so a failure costs minutes and not hours.
