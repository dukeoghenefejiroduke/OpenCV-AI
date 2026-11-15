import os
import cv2
import numpy as np

photos_dir = "/data/data/com.termux/files/home/storage/downloads/testphotos"
triage_dir = "data/triage"
os.makedirs(triage_dir, exist_ok=True)

target_files = ["blur.jpeg", "blue2.jpeg"]

for fname in target_files:
    img_path = os.path.join(photos_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        continue
    
    h, w = img.shape[:2]
    total_pixels = h * w

    # Mean BGR before balancing
    b_ch, g_ch, r_ch = cv2.split(img)
    b_mean, g_mean, r_mean = float(np.mean(b_ch)), float(np.mean(g_ch)), float(np.mean(r_ch))
    avg_mean = (b_mean + g_mean + r_mean) / 3.0

    if b_mean > 0 and g_mean > 0 and r_mean > 0:
        b_gain = avg_mean / b_mean
        g_gain = avg_mean / g_mean
        r_gain = avg_mean / r_mean
        balanced_b = cv2.multiply(b_ch, b_gain)
        balanced_g = cv2.multiply(g_ch, g_gain)
        balanced_r = cv2.multiply(r_ch, r_gain)
        balanced = cv2.merge([balanced_b, balanced_g, balanced_r])
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    else:
        balanced = img.copy()

    b_bal_mean, g_bal_mean, r_bal_mean = float(np.mean(balanced[:,:,0])), float(np.mean(balanced[:,:,1])), float(np.mean(balanced[:,:,2]))

    # Lab conversion + CLAHE
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_lab = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_clahe, a_ch, b_lab])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

    a_float = a_ch.astype(np.float32)
    b_float = b_lab.astype(np.float32)
    a_mean, a_std = np.mean(a_float), np.std(a_float)
    b_mean_val, b_std = np.mean(b_float), np.std(b_float)

    lab_rust_mask = (a_float > (a_mean + 0.5 * max(a_std, 2.0))) & (b_float > (b_mean_val + 0.3 * max(b_std, 2.0)))

    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_min = 20 if np.mean(v_ch) < 60 else 40
    hsv_rust_mask = (h_ch >= 3) & (h_ch <= 32) & (s_ch >= 35) & (v_ch >= v_min)

    combined_mask = (lab_rust_mask & hsv_rust_mask).astype(np.uint8) * 255

    lab_pass_share = (np.sum(lab_rust_mask) / total_pixels) * 100.0
    hsv_pass_share = (np.sum(hsv_rust_mask) / total_pixels) * 100.0
    combined_pass_share = (np.count_nonzero(combined_mask) / total_pixels) * 100.0

    print(f"=== {fname} ===")
    print(f"Before balancing mean BGR: [B: {b_mean:.2f}, G: {g_mean:.2f}, R: {r_mean:.2f}]")
    print(f"After balancing mean BGR:  [B: {b_bal_mean:.2f}, G: {g_bal_mean:.2f}, R: {r_bal_mean:.2f}]")
    print(f"Share passing Lab threshold: {lab_pass_share:.3f}%")
    print(f"Share passing HSV threshold: {hsv_pass_share:.3f}%")
    print(f"Share passing Combined mask: {combined_pass_share:.3f}%")

    cv2.imwrite(os.path.join(triage_dir, f"balanced_{fname}"), balanced)
    cv2.imwrite(os.path.join(triage_dir, f"mask_{fname}"), combined_mask)
