# Twitter-Crawling-NLP-KMeans 🐦📊

A Twitter/X data analysis project using **Natural Language Processing (NLP)**, **Sentiment Analysis**, and **K-Means Clustering**.

## 📌 Overview

This project analyzes Twitter/X data collected through crawling with the keyword **"kapolri"**.

The collected data is processed through text preprocessing, TF-IDF feature extraction, sentiment analysis, and K-Means clustering. The results are then presented through data visualizations and an interactive **Streamlit dashboard**.

Users can also enter their own text to get a sentiment prediction.

## ✨ Features

| Feature                      | Description                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| 🔍 **Sentiment Prediction**  | Predicts text sentiment as positive, negative, or neutral using Logistic Regression. |
| 📊 **Sentiment Analysis**    | Displays the sentiment distribution of the collected dataset.                        |
| ☁️ **Word Cloud**            | Shows frequently occurring words in the dataset.                                     |
| 🔤 **Most Frequent Words**   | Displays the 12 most frequently used words.                                          |
| 📈 **K-Means Clustering**    | Groups text data based on similarity using TF-IDF and 8 clusters.                    |
| 📐 **Clustering Evaluation** | Uses Silhouette Score and PCA to evaluate and visualize the clustering results.      |

## 🛠️ Technologies

* **Python**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **WordCloud**
* **Joblib**
* **Streamlit**

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/twitter-crawling-nlp-kmeans.git
cd twitter-crawling-nlp-kmeans
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Start the Streamlit dashboard with:

```bash
streamlit run dashboard.py
```

Then open the local URL displayed in the terminal to access the dashboard.

## 📂 Dataset

The dataset contains Twitter/X posts collected using the keyword **"kapolri"**.

The data goes through preprocessing, sentiment prediction, and clustering before being displayed in the dashboard.

## 🔄 Analysis Workflow

```text
Twitter/X Data
      │
      ▼
   Crawling
      │
      ▼
 Preprocessing
      │
      ▼
     TF-IDF
      │
      ├───────────────┐
      ▼               ▼
Sentiment        K-Means
 Analysis        Clustering
      │               │
      └───────┬───────┘
              ▼
      Data Visualization
              │
              ▼
      Streamlit Dashboard
```

---

**Developed as part of an NLP and Machine Learning project.**
