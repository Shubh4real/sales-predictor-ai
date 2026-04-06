# 📈 Sales Predictor AI

> An end-to-end AI-powered retail sales forecasting application built with Python, scikit-learn, Streamlit, and Google Gemini API.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Live Demo

**[Launch App →](https://sales-predictor-ai-emcrhafspnexyrmbhlmjap.streamlit.app/)**

---

## 📋 Overview

Sales Predictor AI is a machine learning web application that forecasts retail sales using historical data and provides AI-generated plain-English business insights for non-technical stakeholders.

The project implements a complete ML pipeline — from data ingestion and cleaning through to model training, evaluation, and deployment — and integrates Google Gemini API to automatically explain predictions in a business-friendly format.

**Dataset:** [Kaggle Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) — 3 million rows across 54 stores (2013–2017)

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | **0.923** |
| Mean Absolute Error (MAE) | **1,086 units** |
| Cross-Validation R² (5-fold) | **0.9221** |
| CV Standard Deviation | **0.0023** |
| Train/Test Gap | **0.034** (no overfitting) |

---

## ✨ Features

- **📊 Explore Data** — Interactive charts for daily trends, sales by store, product family, monthly patterns, and promotion impact
- **🔮 Single Prediction** — Select store, date, and parameters to forecast sales with a gauge chart and comparison visualisation
- **🤖 AI Explanation** — Google Gemini generates a structured business insight report from each prediction
- **📂 Batch Prediction** — Upload a test CSV to generate predictions for all rows at once with downloadable results
- **📥 Download Reports** — Export AI-generated insights and batch predictions as CSV/text files

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Machine Learning | scikit-learn (Random Forest Regressor) |
| Data Processing | Pandas, NumPy, SciPy |
| Visualisation | Plotly |
| AI Integration | Google Gemini API |
| Environment | Python 3.8+ |

---

## 🧠 ML Pipeline

The model is trained using a full end-to-end pipeline:

```
Raw Data (3M rows)
    │
    ▼
Data Aggregation (daily per store → 90,936 rows)
    │
    ▼
Data Cleaning (remove negatives, check nulls)
    │
    ▼
Outlier Detection & Removal (IQR method → 85,078 rows)
    │
    ▼
Feature Engineering
    ├── Date features: day_of_week, month, year, quarter, week_of_year, is_weekend
    └── Lag features: lag_7, lag_30, rolling_7 (drop NaN → 83,458 rows)
    │
    ▼
Feature Selection (12 features)
    │
    ▼
Train/Test Split (80/20)
    │
    ▼
StandardScaler (fit on train only)
    │
    ▼
Random Forest Regressor (n_estimators=100, max_depth=15)
    │
    ▼
Evaluation: R² = 0.923 | MAE = 1,086 | CV std = 0.0023
```

**Top Features by Importance:**

| Feature | Importance |
|---------|-----------|
| lag_7 | 68.75% |
| rolling_7 | 21.71% |
| day | 2.87% |
| day_of_week | 1.49% |
| store_nbr | 1.13% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Shubh4real/sales-predictor-ai.git
cd sales-predictor-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Create a `.env` file in the root directory:
```
AIzaSyDTtRoAHs89zsszBvsMIozFMyp7gS4znR0
```

**4. Download the dataset**

Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place it in the `data/` folder.

**5. Train the model**
```bash
python model.py
```

**6. Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
sales-predictor-ai/
│
├── app.py                  # Main Streamlit application (5 pages)
├── model.py                # ML pipeline: clean → feature engineer → train → evaluate
├── explore.py              # Data exploration script
├── requirements.txt        # Python dependencies
│
├── artifacts/              # Saved model artifacts
│   ├── model.pkl           # Trained Random Forest model
│   └── scaler.pkl          # Fitted StandardScaler
│
└── data/                   # Dataset directory (not included)
    └── train.csv           # Download from Kaggle (see Quick Start)
```

---

## 📱 App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview and model performance metrics |
| 📊 Explore Data | Upload CSV and visualise sales trends interactively |
| 🔮 Predict Sales | Single prediction with gauge chart and comparison bar chart |
| 🤖 AI Explanation | Gemini-powered plain-English business insight report |
| 📂 Batch Prediction | Upload test CSV, run predictions on all rows, download results |

---

## 💡 How to Use

**Single Prediction:**
1. Go to **Predict Sales** page
2. Select store number, date, and sales history inputs
3. Click **Run Prediction** to see the forecast
4. Navigate to **AI Explanation** and enter your Gemini API key
5. Click **Generate AI Explanation** for a full business insight report

**Batch Prediction:**
1. Go to **Batch Prediction** page
2. Upload your `test.csv` file
3. Click **Run Batch Prediction**
4. View charts by store, date, and day of week
5. Download the full results CSV

---

## ⚙️ Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `AIzaSyDTtRoAHs89zsszBvsMIozFMyp7gS4znR0` | Google Gemini API key for AI explanations | Yes (for AI page) |

Get a free API key at [aistudio.google.com](https://aistudio.google.com) — no credit card required.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scipy
scikit-learn
plotly
google-generativeai
openpyxl
statsmodels
python-dotenv
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shubham Ravi**
- LinkedIn: [linkedin.com/in/shubh4real](https://linkedin.com/in/shubh4real)
- Email: shubhamravi62@gmail.com
- GitHub: [@Shubh4real](https://github.com/Shubh4real)

---

## 📌 Notes

- The dataset is not included due to file size — download from Kaggle (link above)
- A valid Gemini API key is required for the AI Explanation feature
- The model is pre-trained — run `model.py` to retrain on your own data

---

*Built with Python, scikit-learn, Streamlit and Google Gemini API*
