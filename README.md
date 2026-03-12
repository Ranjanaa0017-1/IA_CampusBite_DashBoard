# 🍱 CampusBite Analytics Dashboard

> A data science dashboard validating the CampusBite meal subscription business model using machine learning and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Business Idea

**CampusBite** is a subscription-based meal planning and delivery platform built for university students. It solves three core student pain points:

| Pain Point | CampusBite Solution |
|-----------|-------------------|
| 💸 High food costs | Affordable weekly meal subscriptions (Basic / Standard / Premium) |
| ⏱️ No time to cook | On-campus delivery within 30 minutes |
| 🥗 Poor nutrition | Customizable dietary plans (Veg, Vegan, Halal, High-Protein) |

---

## 🎯 Project Objective

This dashboard **validates the CampusBite business model** using a survey dataset of 1,500 university students and four machine learning algorithms to answer:

- *Which students are most likely to subscribe?*
- *What segments exist in the student population?*
- *What meal combinations are frequently ordered together?*
- *How much are students willing to pay per month?*

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Business overview and project objectives |
| 🔍 **Dataset Explorer** | Data preview, shape, missing values, summary statistics |
| 📊 **Data Visualizations** | 8 interactive Plotly charts across demographics and behaviour |
| 🤖 **Classification Model** | Random Forest predicting subscription likelihood |
| 👥 **Clustering** | K-Means segmentation into 4 student personas |
| 🔗 **Association Rules** | Apriori algorithm for meal combo discovery |
| 📈 **Regression Forecast** | Linear Regression predicting monthly spending |
| 💡 **Business Insights** | Strategic recommendations from data findings |

---

## 🧠 Algorithms Used

### 1. Classification — Random Forest
- **Goal:** Predict `subscription_interest` (Yes/No)
- **Features:** meal_skip_freq, current_satisfaction, food_budget, distance_to_food, wtp_per_month, and more
- **Output:** Accuracy score, confusion matrix, feature importance

### 2. Clustering — K-Means
- **Goal:** Identify distinct student customer segments
- **Features:** food_budget, wtp_per_month, meal_skip_freq, nutrition_importance, current_satisfaction
- **Output:** Elbow curve, scatter plot, 4 persona profiles

### 3. Association Rule Mining — Apriori
- **Goal:** Discover relationships between dietary preferences, cuisines, and add-ons
- **Library:** `mlxtend`
- **Output:** Rules with support, confidence, and lift metrics

### 4. Regression — Linear Regression
- **Goal:** Forecast `wtp_per_month` (willingness to pay)
- **Features:** food_budget, meal_skip_freq, current_satisfaction, nutrition_importance, distance_to_food
- **Output:** R² score, actual vs predicted plot, residuals

---

## 🗂️ Dataset

The dashboard accepts **CampusBite_Dataset.xlsx** — a synthetic survey dataset of 1,500 university students with 21 columns covering:

- Demographics (age, gender)
- Food habits (cooking_habit, meal_skip_freq, meals_per_day)
- Preferences (diet_type, cuisine_preference, meal_timing)
- Financial (food_budget, wtp_per_month)
- Behaviour (current_app_usage, addon_preference, payment_method)
- Outcomes (subscription_interest, current_satisfaction, referral_likelihood)

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/campusbite-dashboard.git
cd campusbite-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

Upload `CampusBite_Dataset.xlsx` using the sidebar file uploader to unlock all pages.

---

## ☁️ Deploy on Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial CampusBite dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/campusbite-dashboard.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub account** and select the repository

4. **Set the main file path** to `app.py`

5. **Click Deploy** — Streamlit Cloud will auto-install from `requirements.txt`

6. **Upload your dataset** via the sidebar after the app loads

> ⚠️ Do **not** commit `CampusBite_Dataset.xlsx` to GitHub if it contains sensitive data. The sidebar file uploader keeps data in-memory per session only.

---

## 📁 Project Structure

```
campusbite-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Files to exclude from Git
└── .streamlit/
    └── config.toml           # Streamlit theme configuration
```

---

## 🏫 Academic Context

This project was developed as part of a **Project-Based Learning (PBL)** assignment for a Data Science course. The business idea, dataset, and analytical approach were independently developed to demonstrate applied machine learning in a real-world entrepreneurial context.

---

## 📄 License

MIT License — free to use for educational purposes.
