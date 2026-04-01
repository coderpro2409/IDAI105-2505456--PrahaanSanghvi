# Streamlit App Link



# 🛍️ Black Friday Sales Intelligence Dashboard

> **InsightMart Analytics** | Data Mining Year 1 Summative Assessment  
> *Beyond Discounts: Data-Driven Black Friday Sales Insights*

---

## 📌 Project Overview

This project applies data mining techniques to a large-scale Black Friday retail dataset to uncover actionable shopping insights. As a Data Analyst at **InsightMart Analytics**, the goal is to analyse customer purchase patterns, segment shoppers, discover product associations, and flag anomalous spending behaviour — all surfaced through an interactive **Streamlit dashboard**.

---

## 🗂️ Project Structure

```
.
├── app.py               # Main Streamlit dashboard application
├── BlackFriday.csv      # Source dataset (537,577 transaction records)
└── README.md            # This file
```

---

## 📊 Dataset

**File:** `BlackFriday.csv`  
**Records:** 537,577 transactions | **Customers:** 5,891 unique users | **Products:** 3,631 unique products

| Column | Description |
|---|---|
| `User_ID` | Unique customer identifier |
| `Product_ID` | Unique product identifier |
| `Gender` | Customer gender (M / F) |
| `Age` | Age group (e.g. 0-17, 18-25, 26-35 …) |
| `Occupation` | Occupation code (0–20) |
| `City_Category` | City tier (A / B / C) |
| `Stay_In_Current_City_Years` | Years living in current city |
| `Marital_Status` | 0 = Single, 1 = Married |
| `Product_Category_1/2/3` | Product category codes |
| `Purchase` | Transaction value in ₹ |

**Preprocessing applied:**
- Missing values in `Product_Category_2` and `Product_Category_3` filled with `0`
- Gender encoded: Male = 0, Female = 1
- Age ordinal-encoded (0-17 → 1 … 55+ → 7)
- `Stay_In_Current_City_Years` value `"4+"` converted to integer `4`
- Purchase normalised using `MinMaxScaler`
- Duplicate records checked and removed

---

## ✨ Dashboard Stages

The app is organised into six navigable stages via the sidebar:

| Stage | Name | Description |
|---|---|---|
| 1 & 2 | **Overview & Data** | Dataset summary, data quality report, preprocessing steps, and distribution charts |
| 3 | **EDA — Exploratory Analysis** | Demographics, product category analysis, city & stay patterns, and a correlation heatmap |
| 4 | **Customer Clustering** | K-Means segmentation with elbow method and silhouette scoring; interactive K selector |
| 5 | **Association Rule Mining** | Custom Apriori implementation; adjustable support and confidence thresholds |
| 6 | **Anomaly Detection** | Z-Score and IQR methods to identify unusually high spenders |
| 7 | **Insights & Recommendations** | Strategic business recommendations summarising all findings |

---

## 🔍 Key Findings

- **Top spenders:** The 51–55 age group has the highest average purchase (₹9,621)
- **Gender gap:** Males spend on average ₹695 more per transaction than females
- **Most popular category:** Category 5 with 148,592 purchases
- **Cross-sell opportunity:** Customers buying Category 7 are ~42% more likely to buy Category 10 (Lift = 1.42)
- **Anomalous buyers:** 207 customers identified as unusually high spenders (Z-Score > 2.5)
- **4 customer segments:** Budget Shoppers, Moderate Spenders, Frequent Buyers, and Premium Buyers

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the App

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

> ⚠️ Make sure `BlackFriday.csv` is in the **same directory** as `app.py`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Data visualisation |
| `scikit-learn` | K-Means clustering, scaling, silhouette scoring |
| `scipy` | Z-Score anomaly detection |
