import streamlit as st
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="centered"
)

# --- Theme Styling ---
# Streamlit handles Light/Dark modes natively via the top-right settings, 
# but we can explicitly inject custom CSS to polish card layouts.
st.markdown("""
    <style>
    .main-card {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(151, 166, 195, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load the SVC Model ---
@st.cache_resource
def load_model():
    # Make sure 'svc_model.pkl' is in the same directory
    with open('svc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'svc_model.pkl' file not found. Please place it in the same directory as app.py.")
    st.stop()

# --- Title and Header ---
st.title("🎓 Student Performance Prediction")
st.markdown("Enter the student details below to classify their performance status using the trained SVC model.")
st.write("---")

# --- Form Inputs ---
st.subheader("📊 Student Metrics & Background")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=10, max_value=100, value=16, step=1)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=168.0, value=15.0, step=0.5)
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    parent_edu = st.selectbox("Parent Education Level", ["Uneducated/Primary", "High School", "Graduate", "Post-Graduate"])

with col2:
    internet = st.selectbox("Has Internet Access?", ["Yes", "No"])
    extracurricular = st.selectbox("Participates in Extracurriculars?", ["Yes", "No"])
    prev_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
    final_score = st.number_input("Final Test Score / Current Average", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

# --- Pre-processing Inputs to match features ---
# Features mapping based on typical numerical encodings:
gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet == "Yes" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0

# Simple categorical mapping for education (0 to 3 scale)
edu_map = {"Uneducated/Primary": 0, "High School": 1, "Graduate": 2, "Post-Graduate": 3}
edu_encoded = edu_map[parent_edu]

# Array ordered matching: gender, age, study_hours_per_week, attendance_rate, parent_education, internet_access, extracurricular, previous_score, final_score
features = np.array([[
    gender_encoded, 
    age, 
    study_hours, 
    attendance, 
    edu_encoded, 
    internet_encoded, 
    extra_encoded, 
    prev_score, 
    final_score
]])

# --- Prediction Action ---
st.write("---")
if st.button("🔮 Predict Performance Classification", type="primary"):
    with st.spinner("Analyzing metrics..."):
        prediction = model.predict(features)[0]
        
        # Display Results
        st.success("🎉 Prediction Completed Successfully!")
        
        # Styling output box dynamically
        st.markdown(f"""
        <div class="main-card" style="text-align: center;">
            <h3>Prediction Result</h3>
            <h1 style="color: #4F46E5;">Class: {prediction}</h1>
        </div>
        """, unsafe_allow_html=True)
