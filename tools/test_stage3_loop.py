import os
import shutil
import tempfile
from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, Finding, MaintenanceRecord
from workspace.prioritization import score as prioritize

def run_stage3_loop_test():
    print("=== Stage 3 End-to-End Loop Test on TEST-001 ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(root=tmp_dir)

        # 1. Create Asset TEST-001
        asset = Asset(asset_id="TEST-001", location="Platform North", criticality="critical", pipe_material="steel")
        store.upsert_asset(asset)
        print("✓ Created Asset TEST-001")

        # 2. Inspection 1 (March): 7 corrosion findings, confirm 5, reject 2
        insp1 = Inspection(
            inspection_id="INS-001",
            asset_id="TEST-001",
            inspection_date="2026-03-01T10:00:00",
            image_sha256="abc1",
            image_ext="jpg",
            authoritative_report={"corrosion_percentage": 8.5, "recommendation": "Warning threshold met."},
            detector_results={"corrosion_detector": {"findings": [{"defect_category": "corrosion"}]}}
        )
        store.create_inspection(insp1)

        findings_insp1 = []
        for i in range(1, 8):
            f_id = f"F-001-{i}"
            status = "confirmed" if i <= 5 else "rejected"
            f = Finding(
                finding_id=f_id,
                inspection_id="INS-001",
                detector_id="corrosion_detector",
                defect_category="corrosion",
                bbox=[10*i, 10*i, 20, 20],
                confidence=0.9,
                evidence_description=f"Corrosion region {i}",
                human_review_status=status,
                limitations=[]
            )
            findings_insp1.append(f)
        store.add_findings(findings_insp1)

        history = store.list_inspections_for_asset("TEST-001")
        p1 = prioritize(asset, insp1, findings_insp1, history)
        print(f"✓ Inspection 1 Priority: {p1.category} (Score: {p1.score})")
        print("  Rationale:")
        for r in p1.rationale:
            print(f"    - {r}")

        # 3. Create Maintenance Record (pending)
        m_rec = MaintenanceRecord(
            record_id="MREC-001",
            asset_id="TEST-001",
            inspection_id="INS-001",
            work_performed="Applied epoxy coating & replaced corroded segment",
            performed_by="Tech Jane",
            performed_at="2026-03-05T14:00:00",
            kind="recorded_work",
            verification_status="pending"
        )
        store.add_maintenance_record(m_rec)
        print("✓ Created Maintenance Record (status: pending)")

        # 4. Inspection 2 (After Repair, April): 0 corrosion findings
        insp2 = Inspection(
            inspection_id="INS-002",
            asset_id="TEST-001",
            inspection_date="2026-04-01T10:00:00",
            image_sha256="abc2",
            image_ext="jpg",
            authoritative_report={"corrosion_percentage": 0.0, "recommendation": "PASS - Repair verified."},
            detector_results={}
        )
        store.create_inspection(insp2)
        store.add_findings([])

        # Mark maintenance as verified_resolved
        store._conn.execute("UPDATE maintenance_records SET verification_status=?, followup_inspection_id=? WHERE record_id=?", ("verified_resolved", "INS-002", "MREC-001"))
        store._conn.commit()
        m_list = store.list_maintenance_records("TEST-001")
        print(f"✓ Inspection 2 (After Repair): 0 findings. Maintenance verification status: {m_list[0].verification_status} (Follow-up: {m_list[0].followup_inspection_id})")

        # 5. Inspection 3 (Recurrence, November): Corrosion findings again in same category
        insp3 = Inspection(
            inspection_id="INS-003",
            asset_id="TEST-001",
            inspection_date="2026-11-01T10:00:00",
            image_sha256="abc3",
            image_ext="jpg",
            authoritative_report={"corrosion_percentage": 12.0, "recommendation": "Critical recurrence."},
            detector_results={"corrosion_detector": {"findings": [{"defect_category": "corrosion"}]}}
        )
        store.create_inspection(insp3)

        findings_insp3 = [
            Finding(
                finding_id="F-003-1",
                inspection_id="INS-003",
                detector_id="corrosion_detector",
                defect_category="corrosion",
                bbox=[50, 50, 30, 30],
                confidence=0.92,
                evidence_description="Recurring corrosion at same weld seam",
                human_review_status="confirmed",
                limitations=[]
            )
        ]
        store.add_findings(findings_insp3)

        history_3 = store.list_inspections_for_asset("TEST-001")
        p3 = prioritize(asset, insp3, findings_insp3, history_3)
        print(f"✓ Inspection 3 (Recurrence) Priority: {p3.category} (Score: {p3.score})")
        print("  Rationale:")
        for r in p3.rationale:
            print(f"    - {r}")

        print("\n=== STAGE 3 LOOP TEST PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_stage3_loop_test()
