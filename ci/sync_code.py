"""
Job 0 - sync-code (the job GitHub Actions triggers through the CAI API)

Pulls the pushed commit into the CAI project so every downstream job runs
exactly that code. Data, features, models and outputs are git-ignored, so
`git reset --hard` never touches them (it does discard uncommitted edits to
tracked files: develop in Git, not in the CAI project).

EXPECTED_GIT_SHA (set by the trigger in the job run's environment) stops the
chain if the branch has already moved on; the newer push runs its own chain.
"""
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


ROOT = _repo_root()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    branch = os.environ.get("GIT_BRANCH", "main")
    git("fetch", "origin", branch)
    git("reset", "--hard", f"origin/{branch}")
    sha = git("rev-parse", "HEAD")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "git_sha.txt").write_text(sha)
    print(f"project now at {sha}")

    expected = os.environ.get("EXPECTED_GIT_SHA")
    if expected and not sha.startswith(expected):
        print(f"Expected commit {expected} but origin/{branch} is {sha} (newer push?). Stopping chain.")
        return 1
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:   # any SystemExit, even 0, reads as failure under the CAI job kernel
        sys.exit(rc)
