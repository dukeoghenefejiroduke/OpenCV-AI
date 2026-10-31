import os
import zipfile
import hashlib

allowlist = [
    "app/cmms.py",
    "app/dashboard.html",
    "app/digital_twin.py",
    "app/forecasting.py",
    "app/main.py",
    "app/validation.py",
    "agent/decision.py",
    "vision/contract.py",
    "vision/corrosion_detector.py",
    "vision/crack_detector.py",
    "vision/surface_defect_detector.py",
    "vision/leakage_detector.py",
    "vision/insulation_detector.py",
    "vision/orchestrator.py",
    "vision/diagnostics.py",
    "vision/inspection.py",
    "vision/spalling.py",
    "vision/stream.py",
    "tools/evaluate.py",
    "tools/generate_diagrams.py",
    "tools/server.py",
    "tools/make_release.py",
    "tools/make_v1_2_release.py",
    "tools/make_v1_3_release.py",
    "tools/verify_release.py",
    "tools/verify_v1_2_release.py",
    "tools/verify_v1_3_release.py",
    "tools/validate_dataset.py",
    "data/field_validation/README.md",
    "data/field_validation/schema.json",
    "data/field_validation/manifest.jsonl",
    "tests/test_advanced_features.py",
    "tests/test_agent_decision.py",
    "tests/test_dashboard.py",
    "tests/test_diagnostics.py",
    "tests/test_inspection.py",
    "tests/test_multi_hazard.py",
    "tests/test_surface_defect_detector.py",
    "tests/test_leakage_detector.py",
    "tests/test_insulation_detector.py",
    "tests/test_cross_hazard_benchmark.py",
    "tests/test_safety_isolation.py",
    "tests/test_robustness.py",
    "tests/test_validation.py",
    "tests/test_field_validation.py",
    "tests/test_measurement_reliability.py",
    "docs/architecture.md",
    "docs/architecture.png",
    "docs/architecture.svg",
    "docs/technical_report.md",
    "docs/operators_guide.md",
    ".github/workflows/build-lambda.yml",
    "README.md",
    "requirements.txt",
    "requirements-lambda.txt",
    "lambda_handler.py",
    ".gitignore",
    "data/pipe_corrosion.jpg"
]

release_dir = "/data/data/com.termux/files/home/.gemini/tmp/cementsight-ai/release"
os.makedirs(release_dir, exist_ok=True)
zip_name = "cementsight-ai-v1.3.0-source.zip"
zip_path = os.path.join(release_dir, zip_name)

print(f"Creating v1.3.0 release archive at {zip_path}...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for item in allowlist:
        if os.path.exists(item):
            zf.write(item, arcname=item)
            print(f"Included: {item}")
        else:
            print(f"WARNING: Allowlist item not found: {item}")

sha256_hash = hashlib.sha256()
with open(zip_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
checksum = sha256_hash.hexdigest()

checksum_path = zip_path + ".sha256"
with open(checksum_path, "w") as f:
    f.write(f"{checksum}  {zip_name}\n")

print(f"v1.3.0 Archive created successfully!")
print(f"Size: {os.path.getsize(zip_path)} bytes")
print(f"SHA-256: {checksum}")
print(f"Checksum saved to {checksum_path}")
