# Data_Science_Project
Product Recommendation as a Binary Classification Problem (Relevant vs Not Relevant)
# Product Relevance Classification Based on Purchase History

## 📌 Project Overview

This project aims to **classify products as relevant or not relevant for a given user** based on their historical behavior on an e-commerce platform. The system leverages **implicit feedback** such as product views, add-to-cart actions, and purchases to predict whether a product is likely to be relevant to a specific user.

The project uses a real-world e-commerce behavior dataset (RetailRocket-style dataset) and applies **machine learning classification techniques** to model user–item relevance.

---

## 🎯 Objectives

* Build a **binary classification model** to predict product relevance.
* Utilize **user behavior data** (views, cart, purchases) as implicit feedback.
* Perform **feature engineering** on user, item, and interaction data.
* Evaluate the model using standard **classification metrics**.
* Compare multiple **machine learning models**.

---

## 📂 Dataset Description

The dataset consists of three main files:

1. **events.csv**

   * visitorId
   * itemId
   * event (view, addtocart, transaction)
   * timestamp
   * transactionId (if applicable)

2. **item_properties.csv**

   * timestamp
   * itemId
   * property
   * value

3. **category_tree.csv**

   * categoryId
   * parentId

### 🔹 Dataset Characteristics

* 4.5 months of user interaction data
* Over 2.7 million events
* More than 400,000 unique items
* Implicit user feedback (no explicit ratings)

---

## 🏷️ Problem Formulation

This is modeled as a **Supervised Binary Classification Problem**.

### Input (X):

* User behavior statistics
* Item popularity
* Category features
* Historical interaction patterns

### Output (y):

| Condition                            | Label            |
| ------------------------------------ | ---------------- |
| User purchased or added item to cart | 1 (Relevant)     |
| User only viewed or did not interact | 0 (Not Relevant) |

---

## ⚙️ Workflow

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Label Generation**
5. **Train / Test Split**
6. **Model Training**
7. **Model Evaluation**
8. **Result Analysis & Visualization**

---

## 🧠 Machine Learning Models Used

* Logistic Regression
* Random Forest
* XGBoost / LightGBM (optional)
* Neural Network (optional for deep learning extension)

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC–AUC
* Confusion Matrix

---

## 🛠️ Technologies Used

* Python 3.x
* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook / VS Code

---


## ✅ Expected Outcomes

* A trained classification model that predicts product relevance.
* Performance comparison between different models.
* Insights into user purchasing behavior.

---

## 🔮 Future Improvements

* Use deep learning models with embeddings.
* Sequence-based models (LSTM, GRU).
* Real-time recommendation system.
* Deployment using Flask or FastAPI.

---

## 👨‍💻 Author

Graduation / Course Project – Machine Learning & Recommender Systems

---

## 📜 License

This project is for educational and research purposes only.
