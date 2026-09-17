# 🛍️ Shopper Spectrum — Customer Segmentation & Product Recommendation System

An end-to-end E-Commerce Analytics system that segments customers using **RFM Analysis + KMeans Clustering** and recommends products using **Item-Based Collaborative Filtering** — deployed as an interactive Streamlit web app.

**🔗 Live App:** [shopper-spectrum-5t2pyrgr2fcbkpwwo3crqs.streamlit.app](https://shopper-spectrum-5t2pyrgr2fcbkpwwo3crqs.streamlit.app/)

---

## 📌 Problem Statement

E-commerce businesses generate massive transaction data but often fail to translate it into targeted marketing and personalized product discovery. This project solves two connected problems:
1. **Who are our customers?** — Segment them into actionable groups (High-Value, Regular, Occasional, At-Risk) so marketing spend can be targeted.
2. **What should we show them next?** — Recommend relevant products based on purchase similarity patterns.

## 📊 Dataset

- **Source:** [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) — UCI Machine Learning Repository
- **Size:** 541,909 transactions | 4,372 unique customers | 3,866 unique products | 38 countries
- **Period:** December 2010 – December 2011
- **Fields:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## 🧠 Approach

### 1. Data Cleaning
- Removed rows with missing `CustomerID`
- Filtered out cancelled invoices (InvoiceNo starting with 'C')
- Removed negative/zero Quantity and UnitPrice records
- Engineered `TotalAmount = Quantity × UnitPrice`

### 2. Customer Segmentation (RFM + KMeans)
- **Recency** — days since last purchase
- **Frequency** — number of unique invoices
- **Monetary** — total amount spent
- Standardized features with `StandardScaler`
- Selected optimal clusters (**K=4**) using the **Elbow Method** and **Silhouette Score**
- **Result:** Silhouette Score of **0.6162**, Inertia of 4092.14
- Labeled clusters: `High-Value`, `Regular`, `Occasional`, `At-Risk`

### 3. Product Recommendation (Collaborative Filtering)
- Built a Customer × Product pivot matrix
- Computed **Cosine Similarity** on the transposed (Product × Product) matrix
- For any input product, returns the **Top 5 most similar products**

### 4. Deployment
Built a 3-module Streamlit app:
- **Dashboard Overview** — key stats, segment distribution, RFM profiles
- **Product Recommendations** — search a product, get top 5 similar items with similarity scores
- **Customer Segmentation** — input RFM values, get predicted segment + business recommendation

## 📈 Key Results

| Metric | Value |
|---|---|
| Number of Clusters (K) | 4 |
| Silhouette Score | 0.6162 |
| High-Value Customers | 217 |
| Occasional Customers | 4,121 |
| Products in Recommendation Engine | 3,866 |

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn (KMeans, StandardScaler, Cosine Similarity)` · `Matplotlib` · `Seaborn` · `Streamlit`

## 📂 Repository Structure

```
shopper-spectrum/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── notebooks/
│   └── Shopper_Spectrum.ipynb # Full analysis: EDA, RFM, clustering, recommendation engine
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_labels.pkl
│   ├── item_similarity.pkl
│   └── product_list.pkl
└── README.md
```

## ▶️ Run Locally

```bash
git clone https://github.com/akankshasinghgit/shopper-spectrum.git
cd shopper-spectrum
pip install -r requirements.txt
streamlit run app.py
```

## 🔍 Limitations & Future Work

- Dataset covers a single year (Dec 2010–Dec 2011); seasonal trends beyond this window aren't captured.
- Data is heavily UK-skewed; segment behavior may not generalize to other regions without re-validation.
- Collaborative filtering is item-based only — a hybrid approach (content + collaborative) could improve cold-start recommendations for new products.
- Next step: add statistical validation (e.g., ANOVA across cluster Monetary values) to confirm segment differences are significant, not just visually distinct.

## 👤 Author

**Akanksha Kumari**
Course: DA/BA (Data Analytics / Business Analytics)
