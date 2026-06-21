import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="AI-Based Hiring Prediction",
    page_icon="🤖",
    layout="centered"
)

DATA_FILE = "AI-Based Hiring Prediction System.csv"
TARGET = "Recruiter Decision"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data["Certifications"] = data["Certifications"].fillna("No Certification")
    return data

@st.cache_resource
def train_model(data):
    features = [
        "Skills",
        "Experience (Years)",
        "Education",
        "Certifications",
        "Job Role",
        "Salary Expectation ($)",
        "Projects Count"
    ]

    X = data[features]
    y = data[TARGET]

    categorical_cols = ["Skills", "Education", "Certifications", "Job Role"]
    numeric_cols = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, pos_label="Hire"),
    }
    return model, features, metrics

try:
    df = load_data()
    model, features, metrics = train_model(df)

    st.markdown(
        """
        <div style='background-color:#f2f2f2; padding:22px; border-radius:10px; text-align:center;'>
            <h1 style='color:#111;'>🤖 AI-Based Hiring Prediction System</h1>
            <p style='color:#444;'>Predict whether a candidate may be hired or rejected based on profile details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        skills = st.selectbox("Select Skills:", sorted(df["Skills"].dropna().unique()))
        education = st.selectbox("Select Education:", sorted(df["Education"].dropna().unique()))
        job_role = st.selectbox("Select Job Role:", sorted(df["Job Role"].dropna().unique()))
        projects = st.number_input("Enter Projects Count:", min_value=0, max_value=50, value=3)

    with col2:
        experience = st.number_input("Enter Experience (Years):", min_value=0, max_value=50, value=2)
        certifications = st.selectbox("Select Certification:", sorted(df["Certifications"].dropna().unique()))
        salary = st.number_input("Enter Salary Expectation ($):", min_value=0, value=50000, step=1000)

    input_df = pd.DataFrame(
        [[skills, experience, education, certifications, job_role, salary, projects]],
        columns=features,
    )

    if st.button("Predict Hiring Decision", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probability))

        if prediction == "Hire":
            st.success(f"✅ Prediction: {prediction}")
        else:
            st.error(f"❌ Prediction: {prediction}")

        st.write("### Prediction Confidence")
        st.write(f"Hire Probability: **{prob_dict.get('Hire', 0) * 100:.2f}%**")
        st.write(f"Reject Probability: **{prob_dict.get('Reject', 0) * 100:.2f}%**")

    with st.expander("Model Information"):
        st.write(f"Dataset rows: **{df.shape[0]}**")
        st.write(f"Dataset columns: **{df.shape[1]}**")
        st.write(f"Accuracy: **{metrics['Accuracy']:.2f}**")
        st.write(f"F1 Score: **{metrics['F1 Score']:.2f}**")

except FileNotFoundError:
    st.error("CSV file missing. Keep this file in the same folder as app.py:")
    st.code(DATA_FILE)
except Exception as e:
    st.error("Something went wrong.")
    st.exception(e)
