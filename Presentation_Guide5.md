# Presentation Guide 5 — Heatmap Generation & Visual Comparison Explanation

## Topic: How the Heatmap Works (for Guide / Viva Questions)

---

## 1. The 3-Panel Composite Image

When Visual Spoofing Analysis runs, the system generates a **side-by-side 3-panel heatmap**:

| Panel | Label | What it shows |
|-------|-------|--------------|
| **Left** | "Captured Suspect Screenshot" | Live screenshot taken from the URL found in the message (using Selenium headless Chrome) |
| **Center** | "Stored Trusted: SBI" | Reference screenshot stored in the trusted database (captured at setup time) |
| **Right** | "Differences (Red = Changed)" | Pixel-level difference overlay — the actual **heatmap** |

---

## 2. Color Meaning — HOT Colormap Scale

The right panel uses **OpenCV's COLORMAP_HOT** — the same scale used in thermal/infrared imaging.

```
BLACK  →  RED  →  ORANGE  →  YELLOW  →  WHITE
  0       Low     Medium      High      Maximum
(identical)                          (completely different)
```

| Color | Difference Level | Meaning |
|-------|-----------------|---------|
| ⬛ **Black / Dark** | 0 (identical) | Pixels match perfectly — same layout, same content |
| 🔴 **Red** | Low difference | Minor variation — slight colour shade or anti-aliasing |
| 🟠 **Orange** | Medium difference | Layout shift or different UI element at this position |
| 🟡 **Yellow / White** | High difference | Completely changed area — text, image, or section differs |

**Key insight for viva:** Dark (black/red) areas in the header or login card region are the **most dangerous** — they confirm the attacker successfully copied the bank's visual identity.

---

## 3. The Math Behind the Heatmap

### Step 1 — Compute SSIM with pixel-level difference map

```python
from skimage.metrics import structural_similarity

ssim_score, diff = structural_similarity(
    trusted_gray, suspect_gray, full=True
)
# diff[i][j] = similarity at pixel (i,j), range: -1.0 to +1.0
# +1.0 = perfectly identical, -1.0 = completely different
```

### Step 2 — Invert to get difference (not similarity)

```python
diff_normalized = ((1 - diff) * 255).astype(np.uint8)
# Now: 0 = identical pixels,  255 = maximally different pixels
```

### Step 3 — Apply HOT colormap

```python
diff_color = cv2.applyColorMap(diff_resized, cv2.COLORMAP_HOT)
# Maps 0→black, 128→orange, 255→white
```

### Step 4 — Blend with suspect image (overlay)

```python
overlay = cv2.addWeighted(suspect_resized, 0.55, diff_color, 0.45, 0)
# 55% original image + 45% heatmap → readable overlay
```

### Step 5 — Compose 3-panel canvas and save

```python
cv2.imwrite("data/temp/diff_heatmap.png", canvas)
```

---

## 4. Spoofing Detection Logic

```python
visual_threat_score = ssim_score          # 0.0 to 1.0
spoofing_detected   = ssim_score >= 0.5   # threshold: 50%
```

| SSIM Score | Interpretation |
|-----------|---------------|
| < 0.50 | Pages look different — no spoofing |
| 0.50 – 0.70 | Moderate similarity — **SPOOFING flagged** |
| > 0.70 | Very high similarity — **strong cloning detected** |

---

## 5. Example from Demo Scan

**Message scanned:**
> "Dear Customer, your SBI account ending 4521 has been credited with Rs. 15,000.
> Check your balance at https://www.onlinesbi.sbi"

**Results:**
- Domain match: `onlinesbi.sbi` → matched to **SBI** trusted screenshot
- pHash Distance: **66** (moderate)
- SSIM Score: **63.5%** → crosses 50% threshold → **SPOOFING DETECTED**
- Heatmap: Dark regions in header/logo confirm visual similarity to SBI's layout

**What to say to guide:**
> *"The SSIM score of 63.5% means the suspect page is 63.5% structurally similar to
> our stored SBI reference. Since this crosses our 50% spoofing threshold, the system
> flags it. The heatmap's dark regions (black/red) in the header area confirm the SBI
> logo and navigation bar are almost identical to the trusted reference — exactly how
> a phishing site would copy a bank's visual identity."*

---

## 6. Why COLORMAP_HOT was Chosen

- **Intuitive** — everyone understands thermal imaging (hot = danger = red)
- **Perceptually uniform** — easy to spot high-difference regions at a glance
- **Contrasts well** on dark UI backgrounds
- Used in medical imaging and security systems for the same reason

---

## 7. What to Say if Asked: "Why blend the heatmap with the suspect image?"

> *"Pure heatmap alone loses spatial context — you can't tell which part of the webpage
> the red zone corresponds to. By blending 55% original + 45% heatmap, examiners can
> immediately see WHICH UI element (header, login button, logo) was cloned, making
> the analysis actionable."*

---

## 8. Summary for Presentation Slide

```
Heatmap Generation Pipeline:
─────────────────────────────────────────────────────
SSIM(trusted, suspect) → per-pixel diff map
    ↓
Invert diff  (similarity → dissimilarity)
    ↓
Apply COLORMAP_HOT  (black=same, red/orange/white=different)
    ↓
Blend with suspect screenshot (55% image + 45% heatmap)
    ↓
Compose 3-panel: [Suspect | Trusted | Heatmap Overlay]
    ↓
Serve via /api/heatmap → shown in frontend modal
─────────────────────────────────────────────────────
Color scale:  ⬛ BLACK = identical  →  🟠 ORANGE = changed  →  ⬜ WHITE = very different
```
