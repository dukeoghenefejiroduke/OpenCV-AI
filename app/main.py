import argparse
import os
import sys
import cv2
import numpy as np
from vision.inspection import inspect_corrosion


def main():
    parser = argparse.ArgumentParser(description="CementSight AI - Corrosion Inspection MVP with ROI Extent Filtering")
    parser.add_argument("image", nargs="?", default=None, help="Path to input industrial equipment image (optional)")
    parser.add_argument("--output-dir", type=str, default="data/outputs", help="Directory to save annotated output images")
    parser.add_argument("--no-roi", action="store_true", help="Disable foreground ROI segmentation (compare against baseline)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

        base_name = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"annotated_{base_name}")
        roi_output_path = os.path.join(args.output_dir, "foreground_roi.jpg")

        print(f"Inspecting image: {image_path} (ROI Segmentation: {not args.no_roi})")
        try:
            report = inspect_corrosion(
                image_path,
                output_annotated_path=output_path,
                output_roi_path=roi_output_path,
                use_roi=not args.no_roi
            )
            
            print("\n=== CEMENTSIGHT AI INSPECTION REPORT ===")
            print(f"Recommendation             : {report['recommendation']}")
            print(f"Raw HSV Mask %             : {report.get('raw_hsv_mask_percentage', 'N/A')}%")
            print(f"Foreground ROI Coverage %  : {report.get('foreground_roi_coverage_percentage', 'N/A')}%")
            print(f"Restricted HSV Mask %      : {report.get('restricted_mask_percentage', 'N/A')}%")
            print(f"Post-Morphology Mask %     : {report.get('post_morph_mask_percentage', 'N/A')}%")
            print(f"Final Corrosion %          : {report['corrosion_percentage']}%")
            print(f"Detected Regions Count     : {report['region_count']}")
            print(f"Dimensions                 : {report['dimensions']['width']}x{report['dimensions']['height']} pixels")
            print(f"Bounding Boxes             : {report['bounding_boxes']}")
            
            roi_total = report.get('roi_components_total', 0)
            roi_audit = report.get('roi_rejection_audit', [])
            print(f"ROI Components Before Filter: {roi_total}")
            print(f"ROI Components Rejected     : {len(roi_audit)}")
            if roi_audit:
                print("Rejected ROI Components Audit:")
                for item in roi_audit:
                    print(f"  - Box: {item.get('box')}, Extent: {item.get('extent')}, Area: {item.get('area')}")

            print(f"Output Image               : {output_path}")
            print(f"Diagnostic ROI Mask        : {roi_output_path}")
            print("=========================================\n")
        except Exception as e:
            print(f"Error during inspection: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No image path supplied. Running synthetic test/demo mode...")
        output_path = os.path.join(args.output_dir, "synthetic_annotated.jpg")
        roi_output_path = os.path.join(args.output_dir, "foreground_roi.jpg")

        syn_img = np.full((300, 300, 3), 180, dtype=np.uint8)
        cv2.rectangle(syn_img, (100, 100), (180, 180), (30, 100, 180), -1)

        report = inspect_corrosion(
            syn_img,
            output_annotated_path=output_path,
            output_roi_path=roi_output_path,
            use_roi=True
        )

        print("\n=== CEMENTSIGHT AI SYNTHETIC REPORT ===")
        print(f"Recommendation             : {report['recommendation']}")
        print(f"Raw HSV Mask %             : {report.get('raw_hsv_mask_percentage', 'N/A')}%")
        print(f"Foreground ROI Coverage %  : {report.get('foreground_roi_coverage_percentage', 'N/A')}%")
        print(f"Restricted HSV Mask %      : {report.get('restricted_mask_percentage', 'N/A')}%")
        print(f"Post-Morphology Mask %     : {report.get('post_morph_mask_percentage', 'N/A')}%")
        print(f"Final Corrosion %          : {report['corrosion_percentage']}%")
        print(f"Detected Regions Count     : {report['region_count']}")
        print(f"Dimensions                 : {report['dimensions']['width']}x{report['dimensions']['height']} pixels")
        print(f"Bounding Boxes             : {report['bounding_boxes']}")
        print(f"Output Image               : {output_path}")
        print(f"Diagnostic ROI Mask        : {roi_output_path}")
        print("=======================================\n")


if __name__ == "__main__":
    main()
