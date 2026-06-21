# 🤖 AI-Based Hiring Prediction System

A Machine Learning web application that predicts whether a job candidate is likely to be **Hired** or **Rejected** based on their profile details, including skills, education, experience, certifications, job role, salary expectation, and project count.

The application is built using **Python**, **Scikit-learn**, and **Streamlit** with a **Random Forest Classifier** to provide real-time hiring predictions.

---

## 🚀 Features

* Interactive Streamlit Web Application
* Candidate Hiring Prediction (Hire / Reject)
* Real-Time Prediction
* Prediction Confidence (Hire & Reject Probability)
* Dynamic Input Fields
* Missing Value Handling
* User-Friendly Interface

---

## 📊 Input Features

The model predicts hiring decisions using the following candidate information:

* Skills
* Experience (Years)
* Education
* Certifications
* Job Role
* Salary Expectation ($)
* Projects Count

---

## 🎯 Output

The application predicts:

* ✅ Hire
* ❌ Reject

It also displays:

* Hire Probability
* Reject Probability

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Missing Value Handling
4. Feature Selection
5. Label Encoding & One-Hot Encoding
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Streamlit Web Deployment

---

## 📂 Project Structure

```text
AI-Based-Hiring-Prediction/
│
├── app.py
├── AI-Based Hiring Prediction System.csv
├── ai_based_hiring.ipynb
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 🤖 Machine Learning Models

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Random Forest Classifier

---

## 🔧 Data Preprocessing

* Missing Value Handling
* Label Encoding
* One-Hot Encoding
* StandardScaler
* Train-Test Split
* Feature Selection

---

## 📈 Model Evaluation

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* GridSearchCV (Hyperparameter Tuning)

---

## 📌 Future Improvements

* Resume PDF Upload
* Resume Parsing using NLP
* Candidate Ranking System
* Resume Skill Extraction
* Recruiter Dashboard
* Interview Recommendation System

---

## 👨‍💻 Author

**Sushant Kumar**

B.Tech – Computer Science & Engineering

Ramgarh Engineering College

---

## 📄 License

This project is created for educational and learning purposes.
