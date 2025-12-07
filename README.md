# SplitTest AI — A/B Testing Intelligence Platform

A Streamlit application for analyzing digital product experiments. Upload A/B test datasets, validate experiment health, run statistical tests, visualize results, compute sample sizes, and export shareable reports.

---

## Table of Contents
- [Key Features](#key-features)
- [Dataset Schema](#dataset-schema)
- [Tech Stack](#tech-stack)
- [Statistical Methods](#statistical-methods)
- [Quick Start (Run Locally)](#quick-start-run-locally)
- [Deployment Options](#deployment-options)
- [Example Output](#example-output)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## Key Features

| Category | Capability |
|---|---|
| 🧪 Core Experiment Analysis | Two-proportion Z-test for CTR and conversion significance |
| 📏 Confidence Reporting | Confidence intervals, effect size, lift %, p-values |
| 🛡 Randomization Checks | Chi-square tests to detect biased assignment across segments |
| 📊 Interactive Dashboards | CTR, purchase rate, and segmentation visualizations (Plotly) |
| 🎯 Sample Size & Power | Computes required traffic to detect target uplift |
| 🧠 Insight Generator | Stakeholder-tailored narrative summaries (PM / Data Scientist / Executive) |
| 🔍 Segment Lift Engine | Identifies subgroups with differential treatment effects |
| 📄 PDF Report Export | Generate shareable experiment analysis PDFs |
| 🧬 Multi-file Support | Works with one or more datasets following schema |

---

## Dataset Schema

Your uploaded dataset must include the required columns below. Optional columns improve segmentation and time-series analysis.

Required columns:

| Column | Type / Values | Description |
|---|---:|---|
| `variant` | string (e.g., "A", "B") | Experiment group assignment |
| `clicked_cta` | 0 / 1 | Whether user clicked primary CTA |
| `purchased` | 0 / 1 | Whether user completed conversion |

Optional columns (recommended):

| Column | Type / Values | Purpose |
|---|---:|---|
| `device` | mobile / desktop / tablet | Device-level segmentation |
| `country` | country code / name | Region-based performance split |
| `traffic_source` | organic / email / ads / social | Channel-level analysis |
| `timestamp` | ISO datetime | Time-series / trending analysis |
| `session_duration` | numeric (seconds) | Additional engagement metric |
| `scroll_pct` | 0-100 | Engagement signal |

Sample datasets are provided in `/data`:
- `sample_dataset_1.csv`
- `strong_uplift.csv`
- `segment_effect.csv`

---

## Tech Stack

- Python 3.8+
- Streamlit (web UI)
- Pandas, NumPy (data handling)
- Statsmodels, SciPy (statistical tests)
- Plotly (interactive plots)
- ReportLab (PDF generation)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Statistical Methods

| Method | Purpose |
|---|---|
| Two-Proportion Z-Test | Compare CTR / conversion between variants |
| Confidence Intervals | Quantify uncertainty in observed metrics |
| Chi-Square Test | Check randomization balance across categorical segments |
| Power Analysis / Sample Size | Compute traffic required to detect uplift with given alpha & power |

---

## Quick Start (Run Locally)

1. Clone repository
```bash
git clone https://github.com/DEN11223344/SplitTest-AI.git
cd SplitTest-AI
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deployment Options

Common deployment targets:
- Streamlit Community Cloud (share.streamlit.io)
- Hugging Face Spaces (Streamlit runtime)
- Docker / Cloud Run

Example: Deploy to Streamlit Community Cloud
1. Push repository to GitHub
2. Go to https://share.streamlit.io
3. Select repository and set entry point to `app.py`
4. Deploy

---

## Example Output

- Decision recommendation (Ship / Hold / Retest)
- CTR & Purchase Rate visualizations
- Segment impact breakdown with lift and significance
- Required sample size and power analysis tables
- Downloadable PDF report
- AI-generated stakeholder explanations (summary tailored for PM / Data Scientist / Executive)

UI preview image: add `assets/preview.png` for a screenshot display in README.

---

## Repository Structure

```
SplitTest-AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── sample_dataset_1.csv
│   ├── strong_uplift.csv
│   └── segment_effect.csv
└── assets/
    └── preview.png
```

---

## Contributing

Contributions, issues, and pull requests are welcome.

- Please open issues for bugs or feature requests.
- For code changes, fork the repo and submit a pull request.
- Include tests or example datasets for reproducibility where applicable.

---

## Author

Piyush Rajesh Balode  
Data Science / Product Analytics  
- LinkedIn: https://www.linkedin.com/in/pbalode11  
- Email: balodepiyush2493@gmail.com

---

## License

If you have a preferred license (MIT, Apache-2.0, etc.), add a `LICENSE` file and update this section. If none chosen, add one before public distribution.

---

If you'd like, I can:
- open a PR updating README.md in the repository,
- add the preview screenshot to assets,
- or generate a concise changelog of edits I made. Which should I do next?
