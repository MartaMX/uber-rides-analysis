# 🚕 Uber Rides Analytics

An end-to-end data analytics project that transforms **150,000 ride-booking records from Bengaluru, India** into an interactive decision-support platform.

The project covers the full analytics lifecycle:

**Python → SQL → Streamlit → Gemini API → Machine Learning → Azure**

The goal: turn raw ride data into useful insights for operations, product, and business teams.

---

## 🎯 Project Objectives

* Explore booking demand, revenue, cancellations, and operational performance.
* Build a reusable data cleaning and analytical pipeline.
* Create an interactive dashboard for business decision-making.
* Add AI-generated interpretations of dashboard results.
* Test whether booking outcomes can be predicted with machine learning.
* Deploy the final analytics product to the cloud.

---

## 🗺️ Dataset

The dataset contains **150,000 synthetic ride-booking records from Bengaluru, India**, including:

* Booking date, time, status, and value
* Vehicle type and ride distance
* Pickup and drop-off locations
* Payment method
* Customer and driver ratings
* Arrival times, cancellations, and incomplete rides

> The dataset is synthetic, so observed patterns should not be interpreted as real Uber operational performance.

---

## 🧹 Data Preparation & Feature Engineering

The data was cleaned and validated in Python using **pandas**, including missing-value analysis, duplicate detection, data-type validation, and business-rule checks.

Additional analytical features were engineered for:

* Time and date analysis
* Booking outcomes
* Completed and cancelled rides
* Realized revenue
* Potential revenue loss

The resulting dataset provides a reusable analytical layer for SQL analysis, the dashboard, and machine learning.

---

## 🔎 Exploratory Data Analysis

Exploratory analysis was performed in Jupyter Notebook to understand the dataset before developing the dashboard.

The analysis focused on:

* Booking demand and time patterns
* Revenue and booking value
* Vehicle performance
* Cancellations and their causes
* Pickup and drop-off locations
* Popular routes
* Operational performance

---

## 🗄️ SQL Analytics

The cleaned dataset was loaded into **MySQL** to reproduce and validate key business metrics using SQL.

Main analytical areas included:

* Booking volume and completion
* Revenue
* Cancellation rates
* Vehicle performance
* Location and route performance
* Payment methods

This demonstrates how the analysis can move from notebook exploration toward a structured analytical database workflow.

---

## 📊 Interactive Streamlit Dashboard

The analysis was transformed into an interactive **Streamlit dashboard** where users can filter the data by:

* Booking status
* Vehicle type
* Time of day

The dashboard combines KPIs and visualizations covering **demand, revenue, service performance, locations, routes, and cancellations**.

This allows users to move from static analysis to interactive business exploration.

---

## 🤖 AI-Powered Interpretation

The dashboard integrates the **Google Gemini API** to generate business-oriented interpretations of the currently filtered results.

```text
Dashboard filters
       ↓
Calculated KPIs
       ↓
Structured JSON summary
       ↓
Gemini API
       ↓
Business interpretation
```

The calculations remain deterministic in Python; the LLM is used to **interpret and communicate the results**, rather than calculate them.

---

## 🧠 Machine Learning

Two experiments explored whether booking completion could be predicted:

1. **Request-time model** — uses only information available when a booking is requested.
2. **Experimental model** — includes post-outcome/leakage-prone variables to test whether stronger predictive signals exist later in the ride lifecycle.

The request-time models showed limited predictive power, highlighting an important finding: **the synthetic dataset does not contain strong enough predictive relationships for reliable booking-outcome prediction**.

The modelling pipeline can nevertheless be reused with richer production data.

---

## ☁️ Azure Deployment

The final Streamlit application is deployed using **Microsoft Azure App Service**.

```text
Raw Dataset
     ↓
Python / pandas
     ↓
Clean Analytical Dataset
     ↓
SQL + Analytics
     ↓
Streamlit Dashboard
     ↓
Gemini AI Interpretation
     ↓
Azure App Service
```

API credentials are stored securely as environment variables and are excluded from the GitHub repository.

---

## 🛠️ Technology Stack

| Area             | Technology         |
| ---------------- | ------------------ |
| Programming      | Python             |
| Data Analysis    | pandas, NumPy      |
| Visualization    | Plotly, Matplotlib |
| Analysis         | Jupyter Notebook   |
| Database         | MySQL              |
| Dashboard        | Streamlit          |
| Machine Learning | scikit-learn       |
| Generative AI    | Google Gemini API  |
| Cloud            | Microsoft Azure    |
| Version Control  | Git & GitHub       |

---

## 📁 Project Structure

```text
uber-rides-analysis/
│
├── data/
│   └── clean_uber_rides.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── exploratory_data_analysis.ipynb
│   └── machine_learning.ipynb
│
├── dashboard/
│   └── streamlit_app.py
│
├── sql/
│   └── queries.sql
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Run Locally

Clone the repository:

```bash
git clone <repository-url>
cd uber-rides-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
```

Run the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

---

## 💡 Key Takeaway

This project demonstrates how a dataset can evolve into a complete analytics product:

**Raw Data → Data Validation → EDA → SQL → Interactive Dashboard → AI Interpretation → Machine Learning → Cloud Deployment**

The machine-learning results also reinforce an important lesson: sophisticated algorithms cannot compensate for weak predictive signals in the underlying data.

The value of the project lies not only in the insights from this dataset, but in the **reusable end-to-end architecture** built around it.

---

## 🔮 Next Steps

Future development could include:

* Connecting to live production data
* Automated data ingestion and refresh
* Retraining and monitoring predictive models
* Storing AI interpretations and filter configurations
* Docker containerization
* CI/CD deployment
* Automated data-quality monitoring

---

## ⚠️ Disclaimer

This project was developed for educational and portfolio purposes using synthetic data. It is not affiliated with or endorsed by Uber.

---

## 👩‍💻 Author

**Marta Guzmán**
Data Analytics Final Project — Ironhack

**Better data. Better decisions. Better journeys.**
