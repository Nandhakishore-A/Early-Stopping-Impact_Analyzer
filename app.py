import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# LOAD MODELS
model1 = load_model("model1.h5", compile=False)
model2 = load_model("model2.h5", compile=False)

# LOAD SCALER
scaler = joblib.load("scaler.pkl")

# TITLE
st.title("Student Pass/Fail Prediction")

# MODEL SELECTION
selected_model = st.selectbox(
    "Choose Model",
    [
        "Model 1: (Without Using Early Stopping)",
        "Model 2: (With using Early Stopping)"
    ]
)

# CREATE 2 COLUMNS
col1, col2 = st.columns(2)

with col1:

    previous_score = st.slider("Previous Score", 0, 100, 50)

    math_prev_score = st.slider("Math Score", 0, 100, 50)

    science_prev_score = st.slider("Science Score", 0, 100, 50)

    language_prev_score = st.slider("Language Score", 0, 100, 50)

    daily_study_hours = st.slider("Study Hours", 0, 12, 4)

    attendance_percentage = st.slider("Attendance %", 0, 100, 75)

with col2:

    homework_completion_rate = st.slider("Homework %", 0, 100, 70)

    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

    screen_time_hours = st.slider("Screen Time", 0, 12, 4)

    physical_activity_minutes = st.slider("Physical Activity", 0, 120, 30)

    motivation_score = st.slider("Motivation", 0, 10, 5)

    exam_anxiety_score = st.slider("Exam Anxiety", 0, 10, 5)

# PREDICT BUTTON
if st.button("Predict Result"):

    input_data = np.array([[

        previous_score,
        math_prev_score,
        science_prev_score,
        language_prev_score,
        daily_study_hours,
        attendance_percentage,
        homework_completion_rate,
        sleep_hours,
        screen_time_hours,
        physical_activity_minutes,
        motivation_score,
        exam_anxiety_score

    ]])

    # SCALE INPUT
    input_scaled = scaler.transform(input_data)

    # SELECT MODEL
    if "Model 1" in selected_model:
        prediction = model1.predict(input_scaled, verbose=0)
    else:
        prediction = model2.predict(input_scaled, verbose=0)

    probability = prediction[0][0]

    # RESULT
    if probability >= 0.5:

        st.success(
            f"Prediction : PASS\n\nPASS Probability : {probability*100:.2f}%"
        )

    else:

        fail_probability = 1 - probability

        st.error(
            f"Prediction : FAIL\n\nFAIL Probability : {fail_probability*100:.2f}%"
        )