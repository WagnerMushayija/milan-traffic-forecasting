# Comparative Time Series Analysis and Forecasting of Mobile Network Traffic

**ML Techniques I — Formative Assignment**  
**Telecom Italia Mobile (TIM) — City of Milan (2013)**

---

##  Project Overview

This project involves efficient handling of a large-scale mobile traffic dataset (~5GB, 89 million rows), exploratory data analysis, and one-step-ahead forecasting using classical and neural network models.

**Three main tasks completed:**
- **Task 1**: Memory-efficient data loading and optimization
- **Task 2**: Comprehensive EDA (distribution, time series, stationarity, decomposition, ACF/PACF, spatial analysis, anomalies)
- **Task 3**: Forecasting with **SARIMA**, **LSTM**, and **GRU** models on three target areas

---

##  Repository Structure
'''
milan-mobile-traffic-forecasting/
├── data/                          # Parquet files (not pushed to GitHub)
├── notebooks/
│   └── forecasting_internet_milano.ipynb
├── reports/
│   └── Comparative_Time_Series_Analysis_Report.pdf
├── plots/                         # Generated figures
├── requirements.txt
├── README.md
└── .gitignore
'''


---

## Technologies & Requirements

- **Python** 3.10+
- **Key Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `statsmodels`, `scikit-learn`
  - `torch` (LSTM & GRU)
  - `pyarrow` (Parquet support)

Install dependencies:
```bash
pip install -r requirements.txt

### Key Findings

Best performing model: LSTM
Strong daily and weekly seasonality observed
Neural networks (LSTM & GRU) significantly outperformed SARIMA
Main challenges: Near-zero nighttime traffic and pre-Christmas anomalies (Dec 22)

### Deliverables

Full Report → Comparative_Time_Series_Analysis_Report.pdf
Video Demonstration → Watch 6-minute Demo
Complete Code → forecasting_internet_milano.ipynb
