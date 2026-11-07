#!/usr/bin/env python3
"""Write actual passing test count to data/status/tests_passing.json."""
import json, os, re, subprocess, sys

def main():
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        out = e.output
    m = re.search(r"(\d+) passed", out)
    count = int(m.group(1)) if m else 0
    os.makedirs("data/status", exist_ok=True)
    with open("data/status/tests_passing.json", "w") as f:
        json.dump({"tests_passing": count}, f)
    print(f"tests_passing={count}")

if __name__ == "__main__":
    main()