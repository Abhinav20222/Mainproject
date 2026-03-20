# PhishGuard AI — Presentation Guide 4
# Full Scan Demo Messages

> **How to use this guide:**
> 1. Make sure both servers are running:
>    - `python src\api.py` (Flask backend — port 5000)
>    - `npm run dev` (React frontend — port 5173)
>    - `python demo_server.py` (SBI clone — port 8080) ← needed for Demo 4 only
> 2. Open http://localhost:5173 → Click **Full Scan** tab
> 3. Paste the message, set the visual checkbox as shown, click **Launch Full Scan**

---

## ══════════════════════════════════════
## ✅  SAFE MESSAGES (3)
## ══════════════════════════════════════

---

### SAFE DEMO 1 — Normal Bank Transaction Alert
**Visual Spoofing: ☐ UNCHECKED (no URL)**

```
Dear Customer, your SBI account has been credited with INR 5,000.00 on
20-Mar-2026. Available balance: INR 28,450.00. If not done by you, please
contact our 24x7 helpline on 1800-11-2211. Thank you for banking with SBI.
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | LOW (~10–20) | ✅ Safe — no urgency, no phishing link, normal alert |
| URL | — | Not applicable (no URL in message) |
| Visual | — | Not checked |
| **Combined** | **LOW** | **✅ SAFE** |

**What to say:**
> *"This is a normal bank transaction SMS. The model detects
> no urgency keywords, no suspicious URL, no action bait — correctly
> classified as SAFE."*

---

### SAFE DEMO 2 — Legitimate Shopping Notification (with Visual)
**Visual Spoofing: ☑ CHECKED**

```
Your Flipkart order #FL9923417 has been successfully dispatched and
will arrive by March 22, 2026. Track your package at
https://www.flipkart.com/track or call 1800-208-9898.
Thank you for shopping with Flipkart!
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | LOW | ✅ Safe — no urgency, no threat |
| URL | LOW | ✅ Safe — real Flipkart domain, HTTPS |
| Visual | LOW | ✅ NO CLONING — Flipkart page compared against trusted Flipkart screenshot |
| **Combined** | **LOW** | **✅ SAFE** |

**What to say:**
> *"A legitimate Flipkart delivery SMS. The URL analysis checks 28
> structural features — real domain, HTTPS, no brand spoofing.
> The visual channel takes a live screenshot of Flipkart.com and compares
> it to our stored trusted screenshot — result is NO CLONING."*

---

### SAFE DEMO 3 — Google Security Notification (with Visual)
**Visual Spoofing: ☑ CHECKED**

```
Your Google account sign-in was detected from a new Windows device on
20-Mar-2026. If this was you, no action is needed. To review your
account security, visit https://accounts.google.com/security
— Google Security Team
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | LOW | ✅ Safe — informational, no action pressure |
| URL | LOW | ✅ Safe — Google domain, HTTPS, no suspicious features |
| Visual | LOW | ✅ NO CLONING — Google sign-in page matches stored trusted screenshot |
| **Combined** | **LOW** | **✅ SAFE** |

**What to say:**
> *"A real Google security alert. All three channels score safe.
> The visual pipeline live-captures Google's sign-in page and confirms
> it matches our trusted Google screenshot — no spoofing detected."*

---

## ══════════════════════════════════════
## 🔴  PHISHING MESSAGES (3)
## ══════════════════════════════════════

---

### PHISHING DEMO 4 — SBI KYC Scam + Visual Heatmap ⭐ BEST DEMO
**Visual Spoofing: ☑ CHECKED ← IMPORTANT**
**Requires: `python demo_server.py` running on port 8080**

```
URGENT: Your SBI account has been BLOCKED due to suspicious activity.
Verify your KYC NOW or lose access permanently. Call 1800-XXX-XXXX or
visit http://localhost:8080 to prevent account closure!
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | HIGH (~55–65) | 🔴 Urgency + Financial + Action keywords detected |
| URL | HIGH (~70–80) | 🔴 No HTTPS, suspicious domain structure |
| Visual | HIGH (~65) | 🔴 ⚠ SPOOFING — Closest match: SBI (65.4% SSIM) |
| **Combined** | **~66 HIGH** | **🔴 PHISHING + HEATMAP BUTTON** |

**Step by Step:**
1. Paste message above → Check ☑ Visual Spoofing Analysis → Launch Full Scan
2. Wait ~25–30 seconds (Selenium captures live screenshot of localhost:8080)
3. Result shows: Score 66, HIGH, Channels: sms, url, visual
4. Click **"Heatmap"** button → 3-panel comparison opens
5. Point to panels: Left = Suspect (our clone), Center = Trusted SBI, Right = Differences

**What to say:**
> *"All three detection channels triggered. The SMS model detected urgency,
> financial and action keywords. The URL model flagged missing HTTPS and
> suspicious structure. Most importantly — the visual module used Selenium
> headless Chrome to take a real live screenshot of the submitted URL and
> compared it pixel-by-pixel against our 54 trusted site database using
> pHash and SSIM. It identified this page as visually cloning SBI with
> 65.4% structural similarity — and generated this heatmap showing exactly
> where the pages are identical and where they differ."*

---

### PHISHING DEMO 5 — ICICI OTP Verification Scam
**Visual Spoofing: ☐ UNCHECKED (fake URL, not hosted)**

```
ALERT: Your ICICI Bank NetBanking account will be SUSPENDED in 24 hours
due to incomplete KYC verification. Update your credentials immediately
at http://icici.secure-kyc-update.xyz/verify?cust=9182&ref=KYC2024
or call 9876-XXXXXX before your account is permanently locked!
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | HIGH | 🔴 Urgency + Financial + Action + Threat keywords |
| URL | CRITICAL (~85–99) | 🔴 Brand spoof in subdomain, suspicious words, no HTTPS, query params |
| Visual | — | Not checked |
| **Combined** | **HIGH–CRITICAL** | **🔴 PHISHING** |

**Top Risk Features flagged by URL model:**
- `has_brand_in_subdomain` — "icici" appears in subdomain of fake domain
- `has_suspicious_words` — "secure", "verify", "kyc" in URL
- `has_https` = 0 (no HTTPS)
- `num_dots` — 4 dots (complex subdomain chain)

**What to say:**
> *"The URL model flagged this instantly — 'icici' is a brand name appearing
> in the subdomain of a completely different registered domain. This is
> a classic brand spoofing pattern. The 28 structural features also caught:
> no HTTPS, suspicious keywords in the path, and deep query parameters
> used to track victims."*

---

### PHISHING DEMO 6 — Prize/Lottery Scam
**Visual Spoofing: ☐ UNCHECKED (fake URL, not hosted)**

```
Congratulations! You have been SELECTED to receive Rs.50,000 cashback
reward on your SBI account. Claim your prize NOW before the offer expires
at http://sbi-reward.online/claim?winner=true&code=WIN2024
Limited time only! Reply CLAIM or call 1800-XXX-0000 immediately!
```

| Channel | Expected Score | Expected Result |
|---|---|---|
| SMS | CRITICAL (~80–95) | 🔴 Max indicators: urgency + prize bait + financial + action + phone |
| URL | HIGH (~75–85) | 🔴 Fake TLD .online, brand spoof "sbi-reward", query params |
| Visual | — | Not checked |
| **Combined** | **HIGH–CRITICAL** | **🔴 PHISHING** |

**SMS Indicators flagged:**
- Urgency Keywords: 3+ (NOW, immediately, expires)
- Financial Keywords: 2+ (Rs.50,000, account, cashback)
- Action Keywords: 3+ (Claim, Reply, call)

**What to say:**
> *"Classic lottery scam — maximum phishing signals. The SMS classifier
> picked up urgency, financial bait, action pressure and a phone number.
> The URL model detected 'sbi-reward.online' — 'sbi' as a brand name
> used in the domain itself (not just the subdomain), combined with
> suspicious .online TLD and winner-tracking query parameters."*

---

## COMPLETE DEMO FLOW (Recommended Order)

```
Demo 1 (SAFE, no visual)   → Show baseline safe detection
Demo 2 (SAFE, with visual) → Show visual NO CLONING on real site
Demo 3 (SAFE, with visual) → Reinforce: system correctly clears real URLs
         ── TRANSITION: "Now let us look at what happens with phishing" ──
Demo 5 (PHISHING, URL)     → Show URL brand-spoof detection (fast, no wait)
Demo 6 (PHISHING, SMS+URL) → Show SMS + URL combined high score
Demo 4 (PHISHING, VISUAL)  → 🌟 GRAND FINALE: Full 3-channel + Heatmap
```

> **Tip:** Demo 4 is the most impressive because it shows ALL THREE channels
> working together AND produces a visual heatmap. Save it for last.

---

## QUICK REFERENCE — Visual Spoofing Checkbox

| Demo | Message Type | ☑ Check Visual? | Reason |
|---|---|---|---|
| 1 | Safe — Bank Alert | ☐ No | No URL in message |
| 2 | Safe — Flipkart | ☑ Yes | Real hosted URL → shows NO CLONING |
| 3 | Safe — Google | ☑ Yes | Real hosted URL → shows NO CLONING |
| 4 | Phishing — SBI KYC | ☑ **YES** | localhost:8080 → shows SPOOFING + Heatmap |
| 5 | Phishing — ICICI OTP | ☐ No | Fake URL not hosted, Selenium will fail |
| 6 | Phishing — Prize Scam | ☐ No | Fake URL not hosted, Selenium will fail |

---

## IF GUIDE ASKS: "Why is the phishing site on localhost?"

> *"Hosting a real phishing website is a cybercrime under the IT Act 2000.
> For demonstration purposes, we have set up a locally-hosted visual clone
> of the SBI banking portal. The underlying technology is identical — Selenium
> headless Chrome captures a live screenshot of any URL, and the pHash + SSIM
> pipeline compares it against our database of 54 trusted site screenshots.
> In a real deployment scenario, any actively hosted phishing URL would trigger
> the same visual detection pipeline automatically."*
