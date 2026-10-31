import os
import zipfile
import hashlib
import subprocess
import tempfile
import shutil

release_dir = "/data/data/com.termux/files/home/.gemini/tmp/cementsight-ai/release"
zip_name = "cementsight-ai-v1.3.0-source.zip"
zip_path = os.path.join(release_dir, zip_name)
checksum_path = zip_path + ".sha256"

if not os.path.exists(zip_path) or not os.path.exists(checksum_path):
    print(f"v1.3.0 release archive or checksum not found at {release_dir}. Run tools/make_v1_3_release.py first.")
    exit(1)

print("1. Verifying v1.3.0 SHA-256 checksum...")
with open(checksum_path, "r") as f:
    expected_line = f.read().strip()
expected_checksum = expected_line.split()[0]

sha256_hash = hashlib.sha256()
with open(zip_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
actual_checksum = sha256_hash.hexdigest()

if expected_checksum != actual_checksum:
    print(f"Checksum mismatch! Expected {expected_checksum}, got {actual_checksum}")
    exit(1)
print(f"Checksum verified: {actual_checksum}")

print("2. Verifying ZIP integrity and inspecting contents...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    bad_file = zf.testzip()
    if bad_file:
        print(f"ZIP integrity check failed on file: {bad_file}")
        exit(1)
    print(f"Total files in archive: {len(zf.namelist())}")

print("3. Extracting archive into isolated temporary directory for testing...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmpdir)
    
    print("4. Running test suite from extracted copy...")
    test_res = subprocess.run(["python", "-m", "pytest"], cwd=tmpdir, env={**os.environ, "PYTHONPATH": tmpdir})
    if test_res.returncode != 0:
        print("Tests failed in extracted archive copy!")
        exit(1)
    print("Tests passed successfully in extracted copy!")

    print("5. Running evaluation harness from extracted copy...")
    eval_res = subprocess.run(["python", "tools/evaluate.py"], cwd=tmpdir, env={**os.environ, "PYTHONPATH": tmpdir})
    if eval_res.returncode != 0:
        print("Evaluation harness failed in extracted archive copy!")
        exit(1)
    print("Evaluation harness passed successfully in extracted copy!")

print("All v1.3.0 release verification checks passed successfully!")
