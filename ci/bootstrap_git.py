"""
CAI job cxr-bootstrap-git (one-off, private GitHub repo): turns a blank project into
a clone of the repository over SSH with a read-only deploy key, so cxr-00-sync-code
can `git fetch` on every push.

ci/setup_cai.py --deploy-key uploads this file and the private key to
.ssh/cxr_deploy_key in the project, creates this job and runs it. The public key
is a read-only deploy key on the repository (Settings > Deploy keys).
"""
import os
import subprocess
import sys
from pathlib import Path

REMOTE = os.environ.get("CXR_GIT_REMOTE", "git@github.com:partomia/Chest-X-ray-Triage.git")
BRANCH = os.environ.get("GIT_BRANCH", "main")


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = _repo_root()
    ssh = root / ".ssh"
    key = ssh / "cxr_deploy_key"
    if not key.exists():
        print(f"no deploy key at {key}")
        return 1
    key.chmod(0o600)
    ssh.chmod(0o700)
    home_ssh = Path.home() / ".ssh"
    config = (f"Host github.com\n  IdentityFile {key}\n  IdentitiesOnly yes\n"
              f"  StrictHostKeyChecking accept-new\n  UserKnownHostsFile {ssh / 'known_hosts'}\n")
    if home_ssh.resolve() != ssh.resolve():
        home_ssh.mkdir(mode=0o700, exist_ok=True)
    (home_ssh / "config").write_text(config)
    (home_ssh / "config").chmod(0o600)
    if not (root / ".git").exists():
        git(root, "init", "-q")
    remotes = git(root, "remote").split()
    git(root, "remote", "set-url" if "origin" in remotes else "add", "origin", REMOTE)
    git(root, "fetch", "origin", BRANCH)
    git(root, "checkout", "-f", "-B", BRANCH, f"origin/{BRANCH}")
    git(root, "branch", "--set-upstream-to", f"origin/{BRANCH}", BRANCH)
    print(f"project at {git(root, 'rev-parse', 'HEAD')} ({REMOTE} {BRANCH})")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:   # any SystemExit, even 0, reads as failure under the CAI job kernel
        sys.exit(rc)
