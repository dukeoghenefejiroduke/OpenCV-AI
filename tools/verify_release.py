import os
import zipfile
import hashlib
import subprocess
import shutil

release_dir = "/data/data/com.termux/files/home/.gemini/tmp/cementsight-ai/release"
zip_name = "cementsight-ai-v1.1.0-source.zip"
zip_path = os.path.join(release_dir, zip_name)
checksum_path = zip_path + ".sha256"

print("1. Verifying SHA-256 checksum...")
with open(checksum_path, "r") as f:
    expected_line = f.read().strip()
expected_checksum, _ = expected_line.split()

sha256_hash = hashlib.sha256()
with open(zip_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
actual_checksum = sha256_hash.hexdigest()

assert actual_checksum == expected_checksum, f"Checksum mismatch! Expected {expected_checksum}, got {actual_checksum}"
print(f"Checksum verified: {actual_checksum}")

print("2. Verifying ZIP integrity and inspecting contents...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    bad_file = zf.testzip()
    assert bad_file is None, f"Corrupted file in ZIP: {bad_file}"
    namelist = zf.namelist()
    print(f"Total files in archive: {len(namelist)}")
    for name in namelist:
        assert not (name == ".git" or name.startswith(".git/")), f"Found .git directory file in archive: {name}"
        assert not ".env" in name, f"Found .env file in archive: {name}"

print("3. Extracting archive into isolated temporary directory for testing...")
extract_dir = "/data/data/com.termux/files/home/.gemini/tmp/cementsight-ai/extract_test"
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

print("4. Running test suite from extracted copy...")
res = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=extract_dir, env={**os.environ, "PYTHONPATH": extract_dir})
assert res.returncode == 0, f"Tests failed in extracted copy with exit code {res.returncode}"
print("Tests passed successfully in extracted copy!")

print("5. Running evaluation harness from extracted copy...")
res_eval = subprocess.run(["python3", "tools/evaluate.py"], cwd=extract_dir, env={**os.environ, "PYTHONPATH": extract_dir})
assert res_eval.returncode == 0, f"Evaluation harness failed in extracted copy with exit code {res_eval.returncode}"
print("Evaluation harness passed successfully in extracted copy!")

print("All release verification checks passed successfully!")
