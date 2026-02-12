
### 6. The "Golden Question": Why 40% SMS, 45% URL, 15% Visual?

**Guide's Question:** *"On what basis did you assign these specific weights? Why isn't it 33% each?"*

**Best Answer (The "Risk-Based Approach"):**

*"Sir/Ma'am, these weights are not random. They are based on the **Cybersecurity Kill Chain** principle, which prioritizes the **Attack Vector** (the URL) over the **Social Engineering Lure** (the SMS)."*

#### Detailed Justification Table:

| Component | Weight | Reasoning (Why this specific number?) | Reliability |
| :--- | :--- | :--- | :--- |
| **1. URL Detection** | **45% (Highest)** | **The "Weapon":** In 99% of phishing attacks, the URL is the *final destination* where the theft happens. If the URL is malicious, the threat is confirmed. Structural features (IP address, suspicious domains) are **highly reliable indicators**. | **High** (~95% Accuracy) |
| **2. SMS Content** | **40% (High)** | **The "Lure":** The text reveals the *intent* (urgency, threats). However, legitimate banks also send urgent messages. Text alone can have false positives, so it gets slightly less weight than the URL. | **High** (~98% Accuracy) |
| **3. Visual Forensics** | **15% (Supplementary)** | **The "Disguise":** This is a *confirmation step*. Not all phishing sites clone visuals (some are just login forms). Also, technical issues (slow loading, mobile views) can affect screenshots. Therefore, it is used as a **supporting signal**, not a primary decider. | **Medium** (Subject to network/rendering) |

#### The Formula for Success:
We prioritize **Technical Evidence (URL Structure)** > **Semantic Intent (SMS Text)** > **Visual Similarity (Screenshot)**.

*"Assigning 33% equally would be a mistake because a safe URL with slightly urgent text (e.g., a real bank alert) would get a falsely high threat score. Our weighted approach prevents these false positives."*
