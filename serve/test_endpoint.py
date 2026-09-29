"""
Build a request for the model endpoint from a film, or send it.

  python serve/test_endpoint.py data/incoming/x.jpeg --print-request   # paste into the model's Test tab
  python serve/test_endpoint.py data/incoming/x.jpeg                   # POST to CXR_ENDPOINT_URL

CXR_ENDPOINT_URL, CXR_ENDPOINT_ACCESS_KEY (model Settings) and, with
authentication on, CXR_ENDPOINT_API_KEY (User Settings > API Keys, Model API key).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import requests


def request_body(path: str) -> dict:
    """The file bytes unchanged: quality features (size, contrast) are computed on the original film."""
    with open(path, "rb") as f:
        return {"image_b64": base64.b64encode(f.read()).decode()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("film")
    ap.add_argument("--print-request", action="store_true")
    args, _ = ap.parse_known_args()

    body = request_body(args.film)
    if args.print_request:
        print(json.dumps(body))
        return 0
    url = os.environ["CXR_ENDPOINT_URL"]
    headers = {"Content-Type": "application/json"}
    if os.environ.get("CXR_ENDPOINT_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['CXR_ENDPOINT_API_KEY']}"
    r = requests.post(url, json={"accessKey": os.environ["CXR_ENDPOINT_ACCESS_KEY"], "request": body},
                      headers=headers, timeout=120)
    print(r.status_code, json.dumps(r.json(), indent=2))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
