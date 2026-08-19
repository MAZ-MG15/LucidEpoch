# 🌙 LucidEpoch: Clinical Sleep Assessment Framework

**LucidEpoch** is a machine learning-powered diagnostic suite designed to evaluate and predict multi-output sleep disorder risks using wearable health data. It transitions standard sleep metrics into an actionable, clinical-grade dashboard tailored for academic research and diagnostic evaluation.

## ✨ Key Features

* **Midnight Dark Clinical UI**: A professional, distraction-free environment built with Streamlit, custom CSS, and modern glassmorphism design principles.
* **🧠 Explainable AI (XAI)**: Integrated **SHAP** (SHapley Additive exPlanations) values to break down complex machine learning decisions, revealing precisely which biological factors drove a specific diagnosis.
* **📊 Interactive Data Visualization**: Dynamic, responsive `Plotly` charts including minimalist radial progress rings for sleep efficiency, sleep architecture timelines, and comprehensive diagnostic risk spectrums.
* **🗃️ Batch Patient Processing**: Streamlined clinical intake by allowing CSV uploads to process and predict multiple patient records simultaneously.
* **⌚ Wearable Sync Simulation**: Built-in mock data pipeline simulating secure synchronization from popular smartwatches (Apple Watch, Fitbit, Oura Ring, Garmin).
* **📄 Automated PDF Reporting**: One-click generation of comprehensive, downloadable diagnostic reports summarizing patient predictions and risk factors for external review.

## 🛠️ Technology Stack

* **Frontend Framework**: Streamlit (with Custom CSS)
* **Machine Learning**: Scikit-Learn, MultiOutputClassifier
* **Explainability**: SHAP (TreeExplainer)
* **Data Visualization**: Plotly (Graph Objects & Express), Matplotlib
* **Data Handling**: Pandas, SQLite (Patient records)
* **Reporting**: FPDF

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MAZ-MG15/LucidEpoch.git
   cd LucidEpoch
