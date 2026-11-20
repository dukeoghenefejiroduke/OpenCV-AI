# CementSight AI — Mission Three Material Provenance Specification

**Date:** November 18, 2026  
**Scope:** Specification of structured material values, provenance sources, verification statuses, and safety rules.

## 1. Supported Material Classifications
- `steel`: Carbon steel, stainless steel, alloy steel piping and structural tanks.
- `concrete`: Reinforced concrete structures, foundation pads, secondary containment walls.
- `brick`: Refractory brick, masonry walls, protective tile cladding.
- `wood`: Timber supports, skids, wooden operational platforms.
- `insulation`: Jacketed pipe insulation, foam cladding, mineral wool.
- `unknown` / `not_sure`: Unassessed or unverified material surfaces.
- `mixed`: Composite assets or multi-material assemblies.

## 2. Provenance & Verification Model
To prevent unsafe assumptions, every material record maintains:
- **`declared_material`**: The material type selected or entered by the field inspector.
- **`provenance_source`**: `"inspector_declared"` | `"independently_verified"` | `"default_fallback"`.
- **`verification_status`**: `"unverified"` | `"verified_by_supervisor"` | `"material_tested"`.
- **`timestamp`**: ISO-8601 recording timestamp.
- **`identity`**: User ID or inspector role identifier.

## 3. Safety Enforcement Rules
1. **No Automatic Fact Generation:** An inspector selecting "brick" declares a subjective field observation. It does not automatically invoke lab material testing or certify the asset as defect-free.
2. **Conservative Handling of Unknown/Mixed:** Unknown or mixed materials trigger conservative behavior: steel corrosion detection runs as a screening estimate accompanied by mandatory human review flags, preventing unverified suppression.
3. **Multi-Hazard Preservations:** Restricting steel corrosion screening on non-steel surfaces does not suppress structural crack, leakage, or insulation damage detectors.
