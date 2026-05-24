"""
Verified Reversible Data Hiding Framework
-----------------------------------------
Reference implementation for feasibility-verified reversible
data hiding using prediction-error expansion and
checkerboard embedding.

Author: B. Bharathi
Research Scholar, Department of Mathematics
Vignan's Foundation for Science, Technology & Research

Supervisor: Dr. P. Sudam Sekhar
Year: 2026
"""

import os
import cv2
import time
import zlib
import struct
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = os.getcwd()

images_folder = os.path.join(BASE_DIR, "input", "images")
payload_folder = os.path.join(BASE_DIR, "input", "payloads")
output_root = os.path.join(BASE_DIR, "output")

stego_folder = os.path.join(output_root, "stego_images")
cover_folder_copy = os.path.join(output_root, "cover_images")
csv_folder = os.path.join(output_root, "csv_results")

os.makedirs(stego_folder, exist_ok=True)
os.makedirs(cover_folder_copy, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

results_csv_path = os.path.join(csv_folder, "Verified_RDH_full_results.csv")
summary_csv_path = os.path.join(csv_folder, "Verified_RDH_full_summary.csv")

# ============================================================
# CONFIG
# ============================================================
THRESHOLD_CANDIDATES = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]
METHODS = ["directional", "interpolation"]
HEADER_RESERVED_ODD_COORDS = 12000

HEADER_MAGIC = b"ATRD"
HEADER_TOTAL_BYTES = 4 + 4 + 4 + 1 + 4
HEADER_TOTAL_BITS = HEADER_TOTAL_BYTES * 8  # 136 bits

VALID_IMAGE_EXTS = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
VALID_PAYLOAD_EXTS = (".bin",)

# ============================================================
# FINAL EXPERIMENT SETUP
# ============================================================
image_list = [
    "Aerial.tiff",
    "Airplane.tiff",
    "Baboon_gray.png",
    "Barbara_gray.png",
    "Boat_gray.png",
    "Fishing Boat.tiff",
    "lake.png",
    "TANK.tiff",
    "Truck.png"
]

payload_list = [
    "random_payload_10k_bits.bin",
    "random_payload_20k_bits.bin",
    "random_payload_30k_bits.bin",
    "random_payload_40k_bits.bin",
    "random_payload_50k_bits.bin"
]

def load_grayscale_512(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)

def load_payload_bits(payload_path):
    with open(payload_path, "rb") as f:
        payload_bytes = f.read()
    return np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8)).astype(np.uint8)

def safe_stem(filename):
    return os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")

# ============================================================
# BASIC METRICS
# ============================================================
def mse_u8(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr_u8(a, b):
    mse = mse_u8(a, b)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))

def ssim_u8(a, b):
    return float(compare_ssim(a, b))

def ber_bits(a_bits, b_bits):
    if len(a_bits) != len(b_bits):
        return 1.0
    if len(a_bits) == 0:
        return 0.0
    return float(np.mean(a_bits != b_bits))

def entropy_u8(img):
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())

# ============================================================
# CHECKERBOARD COORDS
# ============================================================
def get_checkerboard_coords(h, w, parity):
    coords = []

    # Boundary pixels are excluded so that every embeddable pixel
    # has a valid 4-neighbor predictor context during both embedding
    # and extraction, which helps preserve strict reversibility.
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (i + j) % 2 == parity:
                coords.append((i, j))
    return coords
# ============================================================
# PREDICTORS (FINAL SCI-SAFE VERSION)
# ============================================================

def local_complexity(img, i, j):
    left  = int(img[i, j - 1])
    right = int(img[i, j + 1])
    up    = int(img[i - 1, j])
    down  = int(img[i + 1, j])

    # Enhanced complexity (better justification)
    return (
        abs(left - right) +
        abs(up - down) +
        abs(left - up) +
        abs(right - down)
    )


# ------------------------------------------------------------
# 1. Directional Predictor
# ------------------------------------------------------------
def directional_predict(img, i, j):
    left  = int(img[i, j - 1])
    right = int(img[i, j + 1])
    up    = int(img[i - 1, j])
    down  = int(img[i + 1, j])

    h_pred = (left + right) / 2.0
    v_pred = (up + down) / 2.0

    dh = abs(left - right)
    dv = abs(up - down)

    pred = h_pred if dh <= dv else v_pred

    return int(np.clip(round(pred), 0, 255))


# ------------------------------------------------------------
# 2. Interpolation Predictor (MAIN METHOD)
# ------------------------------------------------------------
def interpolation_predict(img, i, j):
    left  = int(img[i, j - 1])
    right = int(img[i, j + 1])
    up    = int(img[i - 1, j])
    down  = int(img[i + 1, j])

    # Symmetric 4-neighbor interpolation
    pred = (left + right + up + down) / 4.0

    return int(np.clip(round(pred), 0, 255))


# ------------------------------------------------------------
# Predictor Selector
# ------------------------------------------------------------
def get_predictor(method_name):
    if method_name == "directional":
        return directional_predict
    elif method_name == "interpolation":
        return interpolation_predict
    else:
        raise ValueError(f"Unknown predictor: {method_name}")

# ============================================================
# PREDICTOR DIAGNOSTICS (for ablation tables)
# ============================================================
def predictor_diagnostics(img, predictor):
    h, w = img.shape
    errors = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            pred = predictor(img, i, j)
            errors.append(int(img[i, j]) - pred)

    errors = np.array(errors, dtype=np.int32)
    return {
        "pred_mae": float(np.mean(np.abs(errors))),
        "count_m2": int(np.sum(errors == -2)),
        "count_m1": int(np.sum(errors == -1)),
        "count_0": int(np.sum(errors == 0)),
        "count_p1": int(np.sum(errors == 1)),
        "count_p2": int(np.sum(errors == 2)),
    }

# ============================================================
# REVERSIBLE MAPPING
# ============================================================
def map_error_for_embedding(e, bit=None):
    if bit is not None:
        if e == -2:
            return True, (-4 if bit == 0 else -3), 1
        elif e == -1:
            return True, (-2 if bit == 0 else -1), 1
        elif e == 0:
            return True, (0 if bit == 0 else 1), 1
        elif e == 1:
            return True, (2 if bit == 0 else 3), 1

    if e <= -3:
        return True, e - 2, 0
    elif e >= 2:
        return True, e + 2, 0

    return False, None, 0

def inverse_error_for_extraction(e_prime, still_need_bits):
    if still_need_bits:
        if e_prime == -4:
            return True, -2, 0
        elif e_prime == -3:
            return True, -2, 1
        elif e_prime == -2:
            return True, -1, 0
        elif e_prime == -1:
            return True, -1, 1
        elif e_prime == 0:
            return True, 0, 0
        elif e_prime == 1:
            return True, 0, 1
        elif e_prime == 2:
            return True, 1, 0
        elif e_prime == 3:
            return True, 1, 1
        elif e_prime <= -5:
            return True, e_prime + 2, None
        elif e_prime >= 4:
            return True, e_prime - 2, None
        else:
            return False, None, None
    else:
        if e_prime <= -5:
            return True, e_prime + 2, None
        elif e_prime >= 4:
            return True, e_prime - 2, None
        else:
            return False, None, None

# ============================================================
# HEADER
# ============================================================
def pack_header(total_payload_bits, l1_bits, threshold, payload_crc):
    header = bytearray()
    header += HEADER_MAGIC
    header += struct.pack(">I", int(total_payload_bits))
    header += struct.pack(">I", int(l1_bits))
    header += struct.pack(">B", int(threshold))
    header += struct.pack(">I", int(payload_crc))
    return np.unpackbits(np.frombuffer(bytes(header), dtype=np.uint8)).astype(np.uint8)

def unpack_header(header_bits):
    if len(header_bits) != HEADER_TOTAL_BITS:
        raise ValueError("Header bit length mismatch")
    header_bytes = np.packbits(header_bits).tobytes()
    magic = header_bytes[:4]
    total_payload_bits = struct.unpack(">I", header_bytes[4:8])[0]
    l1_bits = struct.unpack(">I", header_bytes[8:12])[0]
    threshold = struct.unpack(">B", header_bytes[12:13])[0]
    payload_crc = struct.unpack(">I", header_bytes[13:17])[0]
    return magic, total_payload_bits, l1_bits, threshold, payload_crc

# ============================================================
# GENERIC EMBED / EXTRACT ON COORDS
# ============================================================
def embed_bits_on_coords(input_img, coords, bits, predictor, start_idx=0, threshold=None):
    """
    Uses current input_img as prediction reference.
    threshold=None -> no complexity gating
    """
    stego = input_img.copy()
    bit_index = int(start_idx)

    peak_candidates = 0
    shifted_candidates = 0
    skipped_oob_peak = 0
    skipped_oob_shift = 0
    processed_coords = 0

    for (i, j) in coords:
        if threshold is not None:
            c = local_complexity(input_img, i, j)
            if c > threshold:
                continue

        processed_coords += 1
        x = int(input_img[i, j])
        pred = predictor(input_img, i, j)
        e = x - pred

        if e in (-2, -1, 0, 1):
            peak_candidates += 1
        elif e <= -3 or e >= 2:
            shifted_candidates += 1

        if bit_index < len(bits):
            bit = int(bits[bit_index])
            can_modify, e_p, consumed = map_error_for_embedding(e, bit)
        else:
            can_modify, e_p, consumed = map_error_for_embedding(e, None)

        if not can_modify:
            continue

        new_val = pred + e_p
        if new_val < 0 or new_val > 255:
            if e in (-2, -1, 0, 1):
                skipped_oob_peak += 1
            elif e <= -3 or e >= 2:
                skipped_oob_shift += 1
            continue

        stego[i, j] = new_val
        bit_index += consumed

    return stego, {
        "embedded_bits": int(bit_index - start_idx),
        "end_index": int(bit_index),
        "peak_candidates": int(peak_candidates),
        "shifted_candidates": int(shifted_candidates),
        "skipped_oob_peak": int(skipped_oob_peak),
        "skipped_oob_shift": int(skipped_oob_shift),
        "processed_coords": int(processed_coords),
    }

def extract_bits_from_coords(stego_img, coords, embedded_bits_to_extract, predictor, threshold=None):
    """
    Exact inverse using current recovered state.
    This works here because of checkerboard parity separation:
      - Stage 2 odd pixels depend on even neighbors only
      - Stage 1 even pixels depend on odd neighbors only
    """
    recovered = stego_img.copy()
    extracted_bits = []

    for (i, j) in coords:
        if threshold is not None:
            c = local_complexity(recovered, i, j)
            if c > threshold:
                continue

        x_s = int(recovered[i, j])
        pred = predictor(recovered, i, j)
        e_p = x_s - pred

        still_need_bits = (len(extracted_bits) < embedded_bits_to_extract)
        is_modified, e, bit = inverse_error_for_extraction(e_p, still_need_bits)

        if not is_modified:
            continue

        restored_val = pred + e
        if restored_val < 0 or restored_val > 255:
            raise ValueError("Recovered pixel out of [0,255] during extraction")

        recovered[i, j] = restored_val

        if bit is not None and len(extracted_bits) < embedded_bits_to_extract:
            extracted_bits.append(bit)

    return recovered, np.array(extracted_bits, dtype=np.uint8)

# ============================================================
# HEADER CAPACITY CHECK
# ============================================================
def count_header_capacity(input_img, reserved_odd_coords, predictor):
    """
    Count actual payload-carrying opportunities for header bits.
    """
    cap = 0
    for (i, j) in reserved_odd_coords:
        x = int(input_img[i, j])
        pred = predictor(input_img, i, j)
        e = x - pred

        if e in (-2, -1, 0, 1):
            ok0, e0, _ = map_error_for_embedding(e, bit=0)
            ok1, e1, _ = map_error_for_embedding(e, bit=1)
            if ok0 and ok1:
                nv0 = pred + e0
                nv1 = pred + e1
                if 0 <= nv0 <= 255 and 0 <= nv1 <= 255:
                    cap += 1
    return cap

# ============================================================
# FULL VERIFIED RDH RUN FOR ONE THRESHOLD
# ============================================================
def run_full_rdh_with_fixed_threshold(img, payload_bits, method_name, threshold, save_stego=False, stego_path=""):
    predictor = get_predictor(method_name)
    h, w = img.shape

    even_coords = get_checkerboard_coords(h, w, parity=0)
    odd_coords = get_checkerboard_coords(h, w, parity=1)

    reserved_odd_coords = odd_coords[:HEADER_RESERVED_ODD_COORDS]
    odd_payload_coords = odd_coords[HEADER_RESERVED_ODD_COORDS:]

    timings = {}
    t_total_0 = time.time()

    # --------------------------------------------------------
    # STAGE 1: even coords
    # --------------------------------------------------------
    t0 = time.time()
    stego_L1, stats_L1 = embed_bits_on_coords(
        input_img=img,
        coords=even_coords,
        bits=payload_bits,
        predictor=predictor,
        start_idx=0,
        threshold=threshold
    )
    timings["stage1_embed_time_sec"] = time.time() - t0
    embedded_bits_L1 = stats_L1["embedded_bits"]

    # --------------------------------------------------------
    # HEADER CAPACITY CHECK (on stego_L1 reserved odd coords)
    # --------------------------------------------------------
    t0 = time.time()
    hdr_cap = count_header_capacity(stego_L1, reserved_odd_coords, predictor)
    timings["header_capacity_time_sec"] = time.time() - t0

    if hdr_cap < HEADER_TOTAL_BITS:
        return {
            "status": "FAILED",
            "failure_reason": f"Insufficient header capacity at threshold {threshold}: {hdr_cap} < {HEADER_TOTAL_BITS}",
            "threshold": threshold,
            "gross_payload_bits": None,
            "net_payload_bits": None,
            "l1_bits": embedded_bits_L1,
            "l2_bits": None,
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": None,
            "stage2_shifted_candidates": None,
            "stage2_skipped_oob_peak": None,
            "stage2_skipped_oob_shift": None,
            **timings
        }

    # --------------------------------------------------------
    # TEMP STAGE 2 TO DETERMINE TRUE GROSS PAYLOAD
    # --------------------------------------------------------
    t0 = time.time()
    stego_tmp_L2, stats_L2_tmp = embed_bits_on_coords(
        input_img=stego_L1,
        coords=odd_payload_coords,
        bits=payload_bits,
        predictor=predictor,
        start_idx=embedded_bits_L1,
        threshold=threshold
    )
    timings["stage2_temp_embed_time_sec"] = time.time() - t0

    embedded_bits_L2_tmp = stats_L2_tmp["embedded_bits"]
    gross_payload_bits_tmp = embedded_bits_L1 + embedded_bits_L2_tmp

    if gross_payload_bits_tmp < len(payload_bits):
        return {
            "status": "FAILED",
            "failure_reason": f"Insufficient full capacity at threshold {threshold}: gross {gross_payload_bits_tmp} < requested {len(payload_bits)}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_tmp,
            "net_payload_bits": gross_payload_bits_tmp - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": embedded_bits_L2_tmp,
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2_tmp["peak_candidates"],
            "stage2_shifted_candidates": stats_L2_tmp["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2_tmp["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2_tmp["skipped_oob_shift"],
            **timings
        }

    # --------------------------------------------------------
    # HEADER BUILD
    # --------------------------------------------------------
    payload_used = payload_bits[:gross_payload_bits_tmp].copy()
    payload_crc = zlib.crc32(np.packbits(payload_used).tobytes()) & 0xffffffff

    header_bits = pack_header(
        total_payload_bits=gross_payload_bits_tmp,
        l1_bits=embedded_bits_L1,
        threshold=threshold,
        payload_crc=payload_crc
    )

    # --------------------------------------------------------
    # HEADER EMBEDDING ON RESERVED ODD COORDS
    # --------------------------------------------------------
    t0 = time.time()
    stego_H, stats_H = embed_bits_on_coords(
        input_img=stego_L1,
        coords=reserved_odd_coords,
        bits=header_bits,
        predictor=predictor,
        start_idx=0,
        threshold=None
    )
    timings["header_embed_time_sec"] = time.time() - t0

    if stats_H["embedded_bits"] != HEADER_TOTAL_BITS:
        return {
            "status": "FAILED",
            "failure_reason": f"Header embedding failed: {stats_H['embedded_bits']}/{HEADER_TOTAL_BITS} bits",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_tmp,
            "net_payload_bits": gross_payload_bits_tmp - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": embedded_bits_L2_tmp,
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2_tmp["peak_candidates"],
            "stage2_shifted_candidates": stats_L2_tmp["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2_tmp["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2_tmp["skipped_oob_shift"],
            **timings
        }

    # --------------------------------------------------------
    # REAL STAGE 2 EMBEDDING AFTER HEADER
    # --------------------------------------------------------
    t0 = time.time()
    stego_final, stats_L2 = embed_bits_on_coords(
        input_img=stego_H,
        coords=odd_payload_coords,
        bits=payload_used,
        predictor=predictor,
        start_idx=embedded_bits_L1,
        threshold=threshold
    )
    timings["stage2_real_embed_time_sec"] = time.time() - t0

    gross_payload_bits_final = embedded_bits_L1 + stats_L2["embedded_bits"]

    if gross_payload_bits_final != gross_payload_bits_tmp:
        return {
            "status": "FAILED",
            "failure_reason": f"Stage 2 changed after header insertion: temp gross={gross_payload_bits_tmp}, final gross={gross_payload_bits_final}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    if gross_payload_bits_final != len(payload_bits):
        return {
            "status": "FAILED",
            "failure_reason": f"Final gross payload {gross_payload_bits_final} != requested {len(payload_bits)}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    total_embed_time_sec = (
        timings["stage1_embed_time_sec"]
        + timings["header_capacity_time_sec"]
        + timings["stage2_temp_embed_time_sec"]
        + timings["header_embed_time_sec"]
        + timings["stage2_real_embed_time_sec"]
    )

    # Save stego if requested
    if save_stego and stego_path:
        cv2.imwrite(stego_path, stego_final)

    # --------------------------------------------------------
    # DECODING
    # --------------------------------------------------------
    t_dec_0 = time.time()

    # 1) Extract header first
    t0 = time.time()
    after_header_extract_img, extracted_header_bits = extract_bits_from_coords(
        stego_img=stego_final,
        coords=reserved_odd_coords,
        embedded_bits_to_extract=HEADER_TOTAL_BITS,
        predictor=predictor,
        threshold=None
    )
    timings["header_extract_time_sec"] = time.time() - t0

    if len(extracted_header_bits) != HEADER_TOTAL_BITS:
        return {
            "status": "FAILED",
            "failure_reason": f"Header extraction failed: {len(extracted_header_bits)}/{HEADER_TOTAL_BITS}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": mse_u8(img, stego_final),
            "psnr": psnr_u8(img, stego_final),
            "ssim": ssim_u8(img, stego_final),
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": float(entropy_u8(stego_final)),
            "embed_time_sec": total_embed_time_sec,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": stego_path if save_stego else "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    try:
        magic, hdr_total_bits, hdr_l1_bits, hdr_threshold, hdr_crc = unpack_header(extracted_header_bits)
    except Exception as e:
        return {
            "status": "FAILED",
            "failure_reason": f"Header unpack failed: {e}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": mse_u8(img, stego_final),
            "psnr": psnr_u8(img, stego_final),
            "ssim": ssim_u8(img, stego_final),
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": float(entropy_u8(stego_final)),
            "embed_time_sec": total_embed_time_sec,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": stego_path if save_stego else "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    if magic != HEADER_MAGIC:
        return {
            "status": "FAILED",
            "failure_reason": f"Header magic mismatch: got {magic}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": mse_u8(img, stego_final),
            "psnr": psnr_u8(img, stego_final),
            "ssim": ssim_u8(img, stego_final),
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": float(entropy_u8(stego_final)),
            "embed_time_sec": total_embed_time_sec,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": stego_path if save_stego else "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    # header consistency
    if hdr_total_bits != gross_payload_bits_final or hdr_l1_bits != embedded_bits_L1 or hdr_threshold != threshold:
        return {
            "status": "FAILED",
            "failure_reason": f"Header consistency mismatch: total={hdr_total_bits}, l1={hdr_l1_bits}, T={hdr_threshold}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": mse_u8(img, stego_final),
            "psnr": psnr_u8(img, stego_final),
            "ssim": ssim_u8(img, stego_final),
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": float(entropy_u8(stego_final)),
            "embed_time_sec": total_embed_time_sec,
            "extract_time_sec": None,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": stego_path if save_stego else "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    # 2) Extract Stage 2
    l2_bits_expected = hdr_total_bits - hdr_l1_bits
    t0 = time.time()
    after_L2_extract_img, extracted_L2_bits = extract_bits_from_coords(
        stego_img=after_header_extract_img,
        coords=odd_payload_coords,
        embedded_bits_to_extract=l2_bits_expected,
        predictor=predictor,
        threshold=hdr_threshold
    )
    timings["stage2_extract_time_sec"] = time.time() - t0

    # 3) Extract Stage 1
    t0 = time.time()
    recovered_img, extracted_L1_bits = extract_bits_from_coords(
        stego_img=after_L2_extract_img,
        coords=even_coords,
        embedded_bits_to_extract=hdr_l1_bits,
        predictor=predictor,
        threshold=hdr_threshold
    )
    timings["stage1_extract_time_sec"] = time.time() - t0

    total_extract_time_sec = time.time() - t_dec_0

    # Reconstruct payload in original order: Stage1 bits + Stage2 bits
    extracted_payload_bits = np.concatenate([extracted_L1_bits, extracted_L2_bits])

    if len(extracted_payload_bits) != hdr_total_bits:
        return {
            "status": "FAILED",
            "failure_reason": f"Extracted payload length mismatch: got {len(extracted_payload_bits)}, expected {hdr_total_bits}",
            "threshold": threshold,
            "gross_payload_bits": gross_payload_bits_final,
            "net_payload_bits": gross_payload_bits_final - HEADER_TOTAL_BITS,
            "l1_bits": embedded_bits_L1,
            "l2_bits": stats_L2["embedded_bits"],
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": hdr_cap,
            "payload_match": False,
            "exact_recovery": np.array_equal(img, recovered_img),
            "crc_match": False,
            "ber": 1.0,
            "mse": mse_u8(img, stego_final),
            "psnr": psnr_u8(img, stego_final),
            "ssim": ssim_u8(img, stego_final),
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": float(entropy_u8(stego_final)),
            "embed_time_sec": total_embed_time_sec,
            "extract_time_sec": total_extract_time_sec,
            "total_runtime_sec": time.time() - t_total_0,
            "stego_path": stego_path if save_stego else "",
            "stage1_peak_candidates": stats_L1["peak_candidates"],
            "stage1_shifted_candidates": stats_L1["shifted_candidates"],
            "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
            "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
            "stage2_peak_candidates": stats_L2["peak_candidates"],
            "stage2_shifted_candidates": stats_L2["shifted_candidates"],
            "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
            "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
            **timings
        }

    # CRC check
    extracted_crc = zlib.crc32(np.packbits(extracted_payload_bits).tobytes()) & 0xffffffff
    crc_match = (extracted_crc == hdr_crc)

    payload_exact = np.array_equal(extracted_payload_bits, payload_bits[:hdr_total_bits])
    exact_recovery = np.array_equal(recovered_img, img)
    ber_val = ber_bits(payload_bits[:hdr_total_bits], extracted_payload_bits)

    mse_val = mse_u8(img, stego_final)
    psnr_val = psnr_u8(img, stego_final)
    ssim_val = ssim_u8(img, stego_final)
    entropy_orig = entropy_u8(img)
    entropy_stego = entropy_u8(stego_final)

    status = "SUCCESS" if (payload_exact and exact_recovery and crc_match and ber_val == 0.0) else "FAILED"
    failure_reason = ""
    if not payload_exact:
        failure_reason = "Extracted payload mismatch"
    elif not exact_recovery:
        failure_reason = "Recovered cover mismatch"
    elif not crc_match:
        failure_reason = "CRC mismatch"
    elif ber_val != 0.0:
        failure_reason = "BER not zero"

    return {
        "status": status,
        "failure_reason": failure_reason,
        "threshold": threshold,
        "gross_payload_bits": hdr_total_bits,
        "net_payload_bits": hdr_total_bits - HEADER_TOTAL_BITS,
        "l1_bits": hdr_l1_bits,
        "l2_bits": l2_bits_expected,
        "header_bits": HEADER_TOTAL_BITS,
        "header_capacity_bits": hdr_cap,
        "payload_match": bool(payload_exact),
        "exact_recovery": bool(exact_recovery),
        "crc_match": bool(crc_match),
        "ber": float(ber_val),
        "mse": float(mse_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "entropy_orig": float(entropy_orig),
        "entropy_stego": float(entropy_stego),
        "embed_time_sec": float(total_embed_time_sec),
        "extract_time_sec": float(total_extract_time_sec),
        "total_runtime_sec": float(time.time() - t_total_0),
        "stego_path": stego_path if (save_stego and os.path.exists(stego_path)) else "",
        "stage1_peak_candidates": stats_L1["peak_candidates"],
        "stage1_shifted_candidates": stats_L1["shifted_candidates"],
        "stage1_skipped_oob_peak": stats_L1["skipped_oob_peak"],
        "stage1_skipped_oob_shift": stats_L1["skipped_oob_shift"],
        "stage2_peak_candidates": stats_L2["peak_candidates"],
        "stage2_shifted_candidates": stats_L2["shifted_candidates"],
        "stage2_skipped_oob_peak": stats_L2["skipped_oob_peak"],
        "stage2_skipped_oob_shift": stats_L2["skipped_oob_shift"],
        **timings
    }

# ============================================================
# VERIFIED THRESHOLD SEARCH
# ============================================================
def run_verified_threshold_search_full(img, payload_bits, method_name, image_name, payload_name):
    predictor = get_predictor(method_name)
    diag = predictor_diagnostics(img, predictor)

    t_search_0 = time.time()
    trial_log = []
    accepted_result = None

    for T in THRESHOLD_CANDIDATES:
        stego_name = f"{safe_stem(image_name)}_{method_name}_{safe_stem(payload_name)}.png"
        stego_path = os.path.join(stego_folder, stego_name)

        result = run_full_rdh_with_fixed_threshold(
            img=img,
            payload_bits=payload_bits,
            method_name=method_name,
            threshold=T,
            save_stego=False,
            stego_path=stego_path
        )

        trial_log.append({
            "threshold": T,
            "status": result["status"],
            "failure_reason": result["failure_reason"],
            "gross_payload_bits": result["gross_payload_bits"],
            "net_payload_bits": result["net_payload_bits"],
            "l1_bits": result["l1_bits"],
            "l2_bits": result["l2_bits"],
            "psnr": result["psnr"],
            "ssim": result["ssim"],
            "ber": result["ber"],
            "crc_match": result["crc_match"],
            "exact_recovery": result["exact_recovery"],
        })

        if result["status"] == "SUCCESS":
            # rerun once to save stego
            accepted_result = run_full_rdh_with_fixed_threshold(
                img=img,
                payload_bits=payload_bits,
                method_name=method_name,
                threshold=T,
                save_stego=True,
                stego_path=stego_path
            )
            accepted_result["stego_path"] = stego_path
            break

    search_time_sec = time.time() - t_search_0

    if accepted_result is None:
        accepted_result = {
            "status": "FAILED",
            "failure_reason": "No verified threshold found",
            "threshold": None,
            "gross_payload_bits": None,
            "net_payload_bits": None,
            "l1_bits": None,
            "l2_bits": None,
            "header_bits": HEADER_TOTAL_BITS,
            "header_capacity_bits": None,
            "payload_match": False,
            "exact_recovery": False,
            "crc_match": False,
            "ber": None,
            "mse": None,
            "psnr": None,
            "ssim": None,
            "entropy_orig": float(entropy_u8(img)),
            "entropy_stego": None,
            "embed_time_sec": None,
            "extract_time_sec": None,
            "total_runtime_sec": search_time_sec,
            "stego_path": "",
            "stage1_peak_candidates": None,
            "stage1_shifted_candidates": None,
            "stage1_skipped_oob_peak": None,
            "stage1_skipped_oob_shift": None,
            "stage2_peak_candidates": None,
            "stage2_shifted_candidates": None,
            "stage2_skipped_oob_peak": None,
            "stage2_skipped_oob_shift": None,
        }

    accepted_result.update(diag)
    accepted_result["threshold_search_time_sec"] = float(search_time_sec)
    accepted_result["trial_log_json"] = str(trial_log)
    return accepted_result
def main():
    """Run the complete verified RDH experiment pipeline."""
    images_to_process = (
        image_list
        if image_list
        else sorted([f for f in os.listdir(images_folder) if f.lower().endswith(VALID_IMAGE_EXTS)])
    )
    payloads_to_process = (
        payload_list
        if payload_list
        else sorted([f for f in os.listdir(payload_folder) if f.lower().endswith(VALID_PAYLOAD_EXTS)])
    )

    print("Images to process:", images_to_process)
    print("Payloads to process:", payloads_to_process)
    print("Methods:", METHODS)
    print("Threshold candidates:", THRESHOLD_CANDIDATES)
    print("Header bits:", HEADER_TOTAL_BITS)
    print("-" * 100)

    rows = []

    for image_name in images_to_process:
        image_path = os.path.join(images_folder, image_name)
        img = load_grayscale_512(image_path)
        h, w = img.shape

        # SAVE COVER IMAGE ONCE PER IMAGE
        cover_save_path = os.path.join(cover_folder_copy, image_name)
        if not os.path.exists(cover_save_path):
            cv2.imwrite(cover_save_path, img)

        for payload_name in payloads_to_process:
            payload_path = os.path.join(payload_folder, payload_name)
            payload_bits = load_payload_bits(payload_path)

            for method_name in METHODS:
                print(f"Running: Image={image_name} | Payload={payload_name} | Method={method_name}")

                try:
                    result = run_verified_threshold_search_full(
                        img=img,
                        payload_bits=payload_bits,
                        method_name=method_name,
                        image_name=image_name,
                        payload_name=payload_name
                    )

                    gross_bits = result["gross_payload_bits"]
                    net_bits = result["net_payload_bits"]

                    row = {
                        "image": image_name,
                        "payload_file": payload_name,
                        "method": method_name,
                        "image_h": h,
                        "image_w": w,

                        "requested_bits": int(len(payload_bits)),
                        "requested_bpp": float(len(payload_bits) / (h * w)),

                        "status": result["status"],
                        "failure_reason": result["failure_reason"],

                        "selected_threshold": result["threshold"],

                        "gross_payload_bits": gross_bits,
                        "gross_bpp": (float(gross_bits / (h * w)) if gross_bits is not None else None),

                        "header_bits": result["header_bits"],
                        "header_bpp": float(result["header_bits"] / (h * w)),

                        "net_payload_bits": net_bits,
                        "net_bpp": (float(net_bits / (h * w)) if net_bits is not None else None),

                        "l1_bits": result["l1_bits"],
                        "l2_bits": result["l2_bits"],
                        "header_capacity_bits": result["header_capacity_bits"],

                        "payload_match": result["payload_match"],
                        "exact_recovery": result["exact_recovery"],
                        "crc_match": result["crc_match"],
                        "ber": result["ber"],

                        "mse": result["mse"],
                        "psnr": result["psnr"],
                        "ssim": result["ssim"],

                        "entropy_orig": result["entropy_orig"],
                        "entropy_stego": result["entropy_stego"],

                        "threshold_search_time_sec": result.get("threshold_search_time_sec"),
                        "embed_time_sec": result["embed_time_sec"],
                        "extract_time_sec": result["extract_time_sec"],
                        "total_runtime_sec": result["total_runtime_sec"],

                        "stage1_embed_time_sec": result.get("stage1_embed_time_sec"),
                        "header_capacity_time_sec": result.get("header_capacity_time_sec"),
                        "stage2_temp_embed_time_sec": result.get("stage2_temp_embed_time_sec"),
                        "header_embed_time_sec": result.get("header_embed_time_sec"),
                        "stage2_real_embed_time_sec": result.get("stage2_real_embed_time_sec"),
                        "header_extract_time_sec": result.get("header_extract_time_sec"),
                        "stage2_extract_time_sec": result.get("stage2_extract_time_sec"),
                        "stage1_extract_time_sec": result.get("stage1_extract_time_sec"),

                        "pred_mae": result["pred_mae"],
                        "count_m2": result["count_m2"],
                        "count_m1": result["count_m1"],
                        "count_0": result["count_0"],
                        "count_p1": result["count_p1"],
                        "count_p2": result["count_p2"],

                        "stage1_peak_candidates": result["stage1_peak_candidates"],
                        "stage1_shifted_candidates": result["stage1_shifted_candidates"],
                        "stage1_skipped_oob_peak": result["stage1_skipped_oob_peak"],
                        "stage1_skipped_oob_shift": result["stage1_skipped_oob_shift"],

                        "stage2_peak_candidates": result["stage2_peak_candidates"],
                        "stage2_shifted_candidates": result["stage2_shifted_candidates"],
                        "stage2_skipped_oob_peak": result["stage2_skipped_oob_peak"],
                        "stage2_skipped_oob_shift": result["stage2_skipped_oob_shift"],

                        "stego_path": result["stego_path"],
                        "trial_log_json": result["trial_log_json"],
                    }
                    rows.append(row)

                    print(
                        f"  -> {row['status']} | T={row['selected_threshold']} | "
                        f"Gross={row['gross_payload_bits']} | Net={row['net_payload_bits']} | "
                        f"PSNR={row['psnr']} | SSIM={row['ssim']} | BER={row['ber']} | "
                        f"CRC={row['crc_match']} | TotalTime={row['total_runtime_sec']}"
                    )

                except Exception as e:
                    row = {
                        "image": image_name,
                        "payload_file": payload_name,
                        "method": method_name,
                        "image_h": h,
                        "image_w": w,

                        "requested_bits": int(len(payload_bits)),
                        "requested_bpp": float(len(payload_bits) / (h * w)),

                        "status": "ERROR",
                        "failure_reason": str(e),

                        "selected_threshold": None,

                        "gross_payload_bits": None,
                        "gross_bpp": None,

                        "header_bits": HEADER_TOTAL_BITS,
                        "header_bpp": float(HEADER_TOTAL_BITS / (h * w)),

                        "net_payload_bits": None,
                        "net_bpp": None,

                        "l1_bits": None,
                        "l2_bits": None,
                        "header_capacity_bits": None,

                        "payload_match": None,
                        "exact_recovery": None,
                        "crc_match": None,
                        "ber": None,

                        "mse": None,
                        "psnr": None,
                        "ssim": None,

                        "entropy_orig": None,
                        "entropy_stego": None,

                        "threshold_search_time_sec": None,
                        "embed_time_sec": None,
                        "extract_time_sec": None,
                        "total_runtime_sec": None,

                        "stage1_embed_time_sec": None,
                        "header_capacity_time_sec": None,
                        "stage2_temp_embed_time_sec": None,
                        "header_embed_time_sec": None,
                        "stage2_real_embed_time_sec": None,
                        "header_extract_time_sec": None,
                        "stage2_extract_time_sec": None,
                        "stage1_extract_time_sec": None,

                        "pred_mae": None,
                        "count_m2": None,
                        "count_m1": None,
                        "count_0": None,
                        "count_p1": None,
                        "count_p2": None,

                        "stage1_peak_candidates": None,
                        "stage1_shifted_candidates": None,
                        "stage1_skipped_oob_peak": None,
                        "stage1_skipped_oob_shift": None,
                        "stage2_peak_candidates": None,
                        "stage2_shifted_candidates": None,
                        "stage2_skipped_oob_peak": None,
                        "stage2_skipped_oob_shift": None,

                        "stego_path": "",
                        "trial_log_json": "[]",
                    }
                    rows.append(row)
                    print(f"  -> ERROR: {e}")

                print("-" * 100)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_csv_path, index=False)

    # ============================================================
    # SCI-READY SUMMARY
    # ============================================================
    summary_cols = [
        "image",
        "payload_file",
        "method",
        "requested_bits",
        "status",
        "failure_reason",

        "selected_threshold",
        "gross_payload_bits",
        "net_payload_bits",

        "psnr",
        "ssim",
        "ber",
        "crc_match",

        "embed_time_sec",
        "extract_time_sec",
        "total_runtime_sec",

        "stego_path"
    ]

    summary_cols = [c for c in summary_cols if c in results_df.columns]
    summary_df = results_df[summary_cols].copy()
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved full results to:")
    print(results_csv_path)

    print("\nSaved summary to:")
    print(summary_csv_path)

    print("\nCounts by method and status:")
    print(results_df.groupby(["method", "status"]).size())

    print("\nPreview:")
    print(results_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
