import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Title
# ----------------------------
st.set_page_config(page_title="AI Burnout Risk Analyzer", layout="centered")
st.title("💻 AI-Based Burnout Risk Prediction System")

st.markdown("""
⚠️ **Disclaimer:**  
This tool is for demonstration purposes only.  
It does NOT diagnose mental health conditions.
""")

# ----------------------------
# Generate synthetic data (for demo)
# ----------------------------
np.random.seed(42)
n_samples = 300

data = pd.DataFrame({
    "work_hours": np.random.randint(5, 15, n_samples),
    "sleep_hours": np.random.randint(4, 9, n_samples),
    "stress_level": np.random.randint(1, 11, n_samples),
    "tasks_pending": np.random.randint(0, 40, n_samples)
})

data["burnout_risk"] = (
    (data["work_hours"] > 10) &
    (data["sleep_hours"] < 6) &
    (data["stress_level"] > 7)
).astype(int)

# ----------------------------
# Train Model
# ----------------------------
X = data[["work_hours", "sleep_hours", "stress_level", "tasks_pending"]]
y = data["burnout_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

# ----------------------------
# User Input
# ----------------------------
st.header("Employee Lifestyle Inputs")
work_hours = st.slider("Daily Work Hours", 0, 16, 8)
sleep_hours = st.slider("Average Sleep Hours", 0, 12, 6)
stress_level = st.slider("Stress Level (1=low, 10=high)", 1, 10, 5)
tasks_pending = st.slider("Tasks Pending", 0, 50, 10)

# ----------------------------
# Prediction
# ----------------------------
input_data = np.array([work_hours, sleep_hours, stress_level, tasks_pending]).reshape(1, -1)
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ----------------------------
# Display Results
# ----------------------------
st.subheader("Burnout Risk Assessment")
risk_score = probability * 100
st.metric("Burnout Risk Probability", f"{risk_score:.2f}%")

if risk_score > 70:
    st.error("🔥 High Risk of Burnout")
    st.warning("Recommendation: Reduce workload, increase sleep, and consult HR or a wellness advisor.")
elif risk_score > 40:
    st.warning("⚠️ Moderate Risk of Burnout")
    st.info("Monitor stress levels and ensure work-life balance.")
else:
    st.success("✅ Low Risk of Burnout")
    st.info("Maintain healthy routines.")
