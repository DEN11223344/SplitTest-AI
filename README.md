📊 SplitTest AI — A/B Testing Intelligence Platform

A full-featured Streamlit web application for analyzing digital product experiments.
SplitTest AI allows users to upload A/B test datasets, validate experiment health, perform statistical testing, visualize insights, compute required sample sizes, detect segment lift patterns, generate AI-style summaries, and export business-ready PDF reports.

🚀 Features
Feature Category	Capability
🧪 Core Experiment Analysis	Two-proportion Z-test for CTR and conversion significance
📏 Confidence Reporting	Confidence intervals, effect size, lift %, p-values
🛡 Randomization Checks	Chi-square tests to detect biased assignment across segments
📊 Interactive Dashboards	CTR, purchase rate, and segmentation visualizations via Plotly
🎯 Sample Size & Power Calculator	Computes required traffic to detect a target uplift
🧠 AI-Style Insight Generator	Narrative summary tailored to PM / Data Scientist / Executive
🔍 Segment Lift Engine	Identifies groups where treatment works differently
📄 PDF Report Export	Generates shareable experiment analysis PDFs
🧬 Supports Multiple Uploads	Works with any dataset following the schema
📁 Dataset Schema

Your uploaded dataset must include at least:

Column	Description
variant	Experiment group (A/B)
clicked_cta	0/1 — whether user clicked primary CTA
purchased	0/1 — whether user completed conversion

Optional (enhances insights):

Column	Purpose
device	mobile/desktop/tablet
country	region-based performance split
traffic_source	organic/email/ads/social
timestamp	time-series experiment analysis
other engagement metrics	session duration, scroll %, etc.

Sample datasets are included in /data.

🏗️ Tech Stack

Python

Streamlit

Pandas, NumPy

Statsmodels, SciPy (statistical testing)

Plotly (visuals)

ReportLab (PDF report generation)

🧠 Statistical Methods Used
Method	Purpose
Two-Proportion Z-Test	Compare CTR and conversion between Variant A & B
Confidence Intervals	Estimate uncertainty in observed metrics
Chi-Square Test of Independence	Detect randomization imbalance across segments
Power Analysis / Sample Size Calculation	Determine required data to confidently detect uplift
🖥️ How to Run Locally
1. Clone the repository
git clone https://github.com/<your-username>/SplitTest-AI.git
cd SplitTest-AI

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py


The application will open automatically in your browser at:

http://localhost:8501

🌐 Deployment

SplitTest AI can be deployed on:

Streamlit Community Cloud

HuggingFace Spaces

Docker / Cloud Run (optional)

Example Streamlit deployment:

Push project to GitHub

Go to https://share.streamlit.io

Select repository → set entry point to app.py

Deploy

📄 Example Output

🚦 Decision Recommendation (Ship / Hold / Retest)

📈 CTR & Purchase Rate Visualization

🧬 Segment Impact Breakdown

📑 Downloadable PDF Report

🤖 AI-generated explanation for stakeholders

📷 UI Preview

(Optional — include a screenshot)
Add: assets/preview.png

📂 Repository Structure
SplitTest-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── /data
│   ├── sample_dataset_1.csv
│   ├── strong_uplift.csv
│   └── segment_effect.csv
│
└── /assets
    └── preview.png

🧪 Example Use Cases

Website landing page experiment

Email subject line testing

App onboarding flow optimization

Conversion-rate optimization (CRO)

Ecommerce checkout step experiment

Product UI or button color test

📬 Author

PIYUSH RAJESH BALODE
📍 Data Science / Product Analytics
🔗 LinkedIn: www.linkedin.com/in/pbalode11
📧 Email: balodepiyush2493@gmail.com


Pull requests and contributions are welcome!
If you like the project, please ⭐ star the repo — it helps visibility.