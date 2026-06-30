# UX and Usability Enhancement Plan

This plan outlines the steps to overhaul the LucidEpoch Streamlit dashboard to maximize user-friendliness, understandability, and usability by strictly applying established UX (User Experience) laws and principles.

## Goal Description
The current application provides powerful clinical insights and PDF reporting but can be overwhelming for end-users due to technical jargon, cluttered inputs, and potential for user error (e.g., manual inputs not summing to 100%). We will refactor the UI text, layout, and interaction flows to make it intuitive and foolproof.

## Proposed Changes

### [MODIFY] app.py

**1. Error Prevention & Cognitive Load Reduction (Poka-Yoke & Hick's Law)**
- **Issue:** Users manually entering sleep stages might fail to make them sum to 100%, causing a hard error.
- **Fix:** Remove the `Light sleep` slider. Automatically calculate `light_sleep = 100 - rem_sleep - deep_sleep`. If `rem + deep > 100`, cap them automatically. This completely eliminates the possibility of user error and reduces the number of decisions/inputs.

**2. Information Chunking (Miller's Law)**
- **Issue:** The sidebar has a long list of inputs.
- **Fix:** Group inputs into logical chunks with subheaders:
  - 👤 **Demographics** (Age, Gender)
  - 🛏️ **Sleep Metrics** (Duration, Efficiency)
  - 📊 **Sleep Architecture** (REM, Deep Sleep)

**3. Target Acquisition & Visibility (Fitts's Law)**
- **Issue:** The primary "Predict Risks" button could be more prominent.
- **Fix:** Update all primary interaction buttons (Predict, Sync, Download) to use `use_container_width=True` to create larger, easier-to-click targets.

**4. Understandable Language & System Visibility (Nielsen’s Heuristics)**
- **Issue:** The app uses technical jargon (e.g., "Outputs from the MultiOutputClassifier...", "Querying the shap.TreeExplainer...").
- **Fix:** Translate technical ML terms into plain, patient-friendly English:
  - "MultiOutputClassifier..." ➔ "Comprehensive Risk Analysis"
  - "Querying the shap.TreeExplainer..." ➔ "AI Decision Breakdown: Understand which factors influenced your prediction."
- Add a loading spinner (`st.spinner("Analyzing patient data...")`) when "Predict Risks" is clicked to provide immediate system status feedback.

**5. Progressive Disclosure**
- **Issue:** Too much information at once in the Historical Data tab.
- **Fix:** Use expanders for secondary information or raw data downloads to keep the primary view clean.

## User Review Required

> [!IMPORTANT]
> - By automatically calculating **Light Sleep** based on the REM and Deep Sleep inputs, we remove a slider from the UI. This prevents users from getting math errors. Do you approve of this change?
> - The technical language (like mentioning "SHAP" and "MultiOutputClassifier") will be replaced with simpler terms like "AI Decision Breakdown". Is this acceptable for your target audience, or do you need to keep the academic terminology?

## Verification Plan

### Manual Verification
1. Run `streamlit run app.py`.
2. Ensure the sidebar inputs are logically grouped.
3. Test the manual entry mode: verify that adjusting REM and Deep sleep automatically updates Light sleep and never throws a "sum to 100%" error.
4. Read through the tabs to confirm all technical jargon has been simplified.
5. Click "Predict Risks" to ensure the new loading state and layout work flawlessly.
