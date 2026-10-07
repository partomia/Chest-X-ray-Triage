import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Tests never reach the platform, even from a shell that sourced .env (subprocesses inherit this).
for _var in ("CDSW_PROJECT_ID", "CXR_IMPALA_USER", "CXR_IMPALA_PASSWORD", "CXR_CAI_API_KEY", "CXR_VIZ_API_KEY"):
    os.environ.pop(_var, None)
