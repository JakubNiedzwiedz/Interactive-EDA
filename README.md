# 📊 EDA Explorer

**Interactive Streamlit app for quick and visual exploratory data analysis (EDA) with built-in datasets and upload support.**

---

## 🚀 Overview

EDA Explorer is a simple but powerful tool that allows you to quickly analyze datasets without writing code. You can upload your own data or use built-in datasets and instantly explore distributions, relationships, correlations, and outliers.

The app provides an interactive interface with multiple tabs for different aspects of data analysis.

---

## ✨ Features

- 📁 Upload your own datasets (CSV, Excel, JSON)
- 📊 Built-in example datasets (Iris, Titanic, Penguins, etc.)
- 🔍 Automatic detection of:
  - numerical and categorical variables  
  - missing values  
  - duplicates  
- 📈 Interactive visualizations:
  - histograms + KDE
  - bar charts
  - scatter plots with regression
  - boxplots / violin plots
  - correlation heatmaps
- 📉 Statistical analysis:
  - descriptive statistics
  - skewness & kurtosis
  - Shapiro-Wilk normality test
- 🚨 Outlier detection (Z-score method)
- 🎛️ Sidebar filters for categorical variables

---

## 🖥️ App Structure

The application is divided into 6 main sections:

1. **Overview** – dataset preview and basic metrics  
2. **Statistics** – descriptive stats and missing values  
3. **Distributions** – histograms and category distributions  
4. **Relationships** – scatter plots and boxplots  
5. **Correlations** – correlation matrix heatmap  
6. **Outliers** – detection and visualization  

---

## 🛠️ Tech Stack

- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- SciPy  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/eda-explorer.git
cd eda-explorer
