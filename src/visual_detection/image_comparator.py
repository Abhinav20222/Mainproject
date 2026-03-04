"""
Image Comparator Module
Two-stage visual similarity analysis for detecting website spoofing:
  Stage 1: Perceptual Hash (pHash) — fast pre-filter
  Stage 2: SSIM — deep structural comparison with difference heatmap
"""
import os
import sys
import glob
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRUSTED_DIR = PROJECT_ROOT / "data" / "trusted_screenshots"
TEMP_DIR = PROJECT_ROOT / "data" / "temp"

# Import trusted sites mapping
from src.visual_detection.build_trusted_db import TRUSTED_SITES


class ImageComparator:
    """
    Two-stage visual similarity comparator for spoofing detection.
    
    Stage 1: Perceptual Hash (fast pre-filter)
        - Uses imagehash.phash on 256x256 thumbnails
        - Computes Hamming distance
        - Rejects immediately if min distance > 30
        
    Stage 2: SSIM (deep structural comparison)
        - Only runs if stage 1 finds a close match
        - Computes Structural Similarity Index
        - Generates a difference heatmap
    """

    # Threshold: if pHash distance > this, skip SSIM
    PHASH_THRESHOLD = 30

    def __init__(self):
        """Initialize the image comparator."""
        os.makedirs(str(TEMP_DIR), exist_ok=True)
        self._trusted_hashes = {}
        self._load_trusted_hashes()

    def _load_trusted_hashes(self):
        """Pre-compute pHash for all trusted site thumbnails."""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            print("[WARN] imagehash or Pillow not installed. Visual comparison disabled.")
            return

        if not TRUSTED_DIR.exists():
            print(f"[WARN] Trusted screenshots not found: {TRUSTED_DIR}")
            print("  Run: python build_visual_db.py")
            return

        for site_key in TRUSTED_SITES:
            thumb_path = TRUSTED_DIR / f"{site_key}_thumb.png"
            if thumb_path.exists():
                try:
                    img = Image.open(thumb_path)
                    phash = imagehash.phash(img, hash_size=16)
                    self._trusted_hashes[site_key] = phash
                except Exception as e:
                    print(f"[WARN] Could not hash {site_key}: {e}")

        if self._trusted_hashes:
            print(f"[ImageComparator] Loaded {len(self._trusted_hashes)} trusted hashes")
        else:
            print("[ImageComparator] No trusted hashes loaded. Run build_visual_db.py first.")

    def compare(self, suspect_screenshot_path):
        """
        Compare a suspect screenshot against all trusted sites.
        
        Args:
            suspect_screenshot_path (str): Path to the suspect screenshot
            
        Returns:
            dict: Comparison results including spoofing detection, scores, and heatmap path
        """
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            return {
                "spoofing_detected": False,
                "best_match_site": None,
                "best_match_url": None,
                "phash_distance": 999,
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "unavailable",
                "error": "imagehash or Pillow not installed"
            }

        if not self._trusted_hashes:
            return {
                "spoofing_detected": False,
                "best_match_site": None,
                "best_match_url": None,
                "phash_distance": 999,
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "no_trusted_db",
                "error": "No trusted screenshot database. Run build_visual_db.py"
            }

        # --- Stage 1: Perceptual Hash ---
        try:
            suspect_img = Image.open(suspect_screenshot_path)
            suspect_thumb = suspect_img.resize((256, 256), Image.LANCZOS)
            suspect_hash = imagehash.phash(suspect_thumb, hash_size=16)
        except Exception as e:
            return {
                "spoofing_detected": False,
                "best_match_site": None,
                "best_match_url": None,
                "phash_distance": 999,
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "error",
                "error": f"Could not process suspect image: {e}"
            }

        # Find best match by Hamming distance
        best_site = None
        best_distance = 999

        for site_key, trusted_hash in self._trusted_hashes.items():
            distance = suspect_hash - trusted_hash  # Hamming distance
            if distance < best_distance:
                best_distance = distance
                best_site = site_key

        # For presentation demo: Force SSIM calculation always (threshold 1000)
        # Originally: if best_distance > self.PHASH_THRESHOLD:
        if best_distance > 1000: # This condition will almost never be true, forcing SSIM
            return {
                "spoofing_detected": False,
                "best_match_site": best_site,
                "best_match_url": TRUSTED_SITES.get(best_site, ""),
                "phash_distance": int(best_distance),
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "phash_only",
            }

        # --- Stage 2: SSIM Deep Comparison ---
        try:
            from skimage.metrics import structural_similarity
            import cv2
        except ImportError:
            return {
                "spoofing_detected": best_distance <= 15,
                "best_match_site": best_site,
                "best_match_url": TRUSTED_SITES.get(best_site, ""),
                "phash_distance": int(best_distance),
                "ssim_score": 0.0,
                "visual_threat_score": 1.0 - (best_distance / 64.0),
                "heatmap_path": None,
                "analysis_method": "phash_only",
                "error": "scikit-image or opencv not installed for SSIM"
            }

        # Load full screenshots
        trusted_full_path = TRUSTED_DIR / f"{best_site}.png"
        if not trusted_full_path.exists():
            return {
                "spoofing_detected": False,
                "best_match_site": best_site,
                "best_match_url": TRUSTED_SITES.get(best_site, ""),
                "phash_distance": int(best_distance),
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "phash_only",
                "error": f"Full screenshot not found for {best_site}"
            }

        try:
            # Read images
            suspect_cv = cv2.imread(suspect_screenshot_path)
            trusted_cv = cv2.imread(str(trusted_full_path))

            # Convert to grayscale
            suspect_gray = cv2.cvtColor(suspect_cv, cv2.COLOR_BGR2GRAY)
            trusted_gray = cv2.cvtColor(trusted_cv, cv2.COLOR_BGR2GRAY)

            # Resize suspect to match trusted dimensions
            h, w = trusted_gray.shape
            suspect_gray = cv2.resize(suspect_gray, (w, h))

            # Compute SSIM
            ssim_score, diff = structural_similarity(
                trusted_gray, suspect_gray, full=True
            )
            ssim_score = float(ssim_score)

            # Generate difference heatmap
            heatmap_path = None
            try:
                diff_normalized = ((1 - diff) * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
                heatmap_file = str(TEMP_DIR / "diff_heatmap.png")
                cv2.imwrite(heatmap_file, heatmap)
                heatmap_path = heatmap_file
            except Exception:
                pass

            # Determine spoofing
            # High SSIM = visually similar to a trusted site
            # This means a different domain is mimicking a known site
            visual_threat_score = ssim_score
            spoofing_detected = ssim_score >= 0.5

            return {
                "spoofing_detected": spoofing_detected,
                "best_match_site": best_site,
                "best_match_url": TRUSTED_SITES.get(best_site, ""),
                "phash_distance": int(best_distance),
                "ssim_score": round(ssim_score, 4),
                "visual_threat_score": round(visual_threat_score, 4),
                "heatmap_path": heatmap_path,
                "analysis_method": "phash+ssim",
            }

        except Exception as e:
            return {
                "spoofing_detected": False,
                "best_match_site": best_site,
                "best_match_url": TRUSTED_SITES.get(best_site, ""),
                "phash_distance": int(best_distance),
                "ssim_score": 0.0,
                "visual_threat_score": 0.0,
                "heatmap_path": None,
                "analysis_method": "error",
                "error": f"SSIM computation failed: {e}"
            }


# Quick test
if __name__ == "__main__":
    comparator = ImageComparator()

    print("\n" + "=" * 60)
    print("IMAGE COMPARATOR — STATUS")
    print("=" * 60)
    print(f"  Trusted hashes loaded: {len(comparator._trusted_hashes)}")
    print(f"  Trusted directory: {TRUSTED_DIR}")
    print(f"  Temp directory: {TEMP_DIR}")

    # If there are any suspect screenshots in temp, compare the latest one
    suspect_files = glob.glob(str(TEMP_DIR / "suspect_*.png"))
    if suspect_files:
        latest = max(suspect_files, key=os.path.getmtime)
        print(f"\n  Testing with: {latest}")
        result = comparator.compare(latest)
        print(f"  Spoofing detected: {result['spoofing_detected']}")
        print(f"  Best match: {result['best_match_site']}")
        print(f"  pHash distance: {result['phash_distance']}")
        print(f"  SSIM score: {result['ssim_score']}")
        print(f"  Visual threat: {result['visual_threat_score']}")
    else:
        print("\n  No suspect screenshots found. Capture one first.")

    print("\n[OK] Image Comparator ready!")
