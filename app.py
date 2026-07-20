import streamlit as st
import pickle
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION & THEME TOGGLE
# ==========================================
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="centered"
)

# Sidebar UI Theme Injector
st.sidebar.title("🎨 Interface Customization")
theme_choice = st.sidebar.selectbox(
    "Choose Theme Preset", 
    ["Modern Light", "Deep Charcoal Dark", "Cyberpunk Neon"]
)

# Map UI colors based on selection
if theme_choice == "Modern Light":
    bg_color = "#FFFFFF"
    text_color = "#1F2937"
    card_bg = "rgba(79, 70, 229, 0.05)"
    accent_color = "#4F46E5"
elif theme_choice == "Deep Charcoal Dark":
    bg_color = "#111827"
    text_color = "#F9FAFB"
    card_bg = "rgba(255, 255, 255, 0.05)"
    accent_color = "#6366F1"
else: # Cyberpunk Neon
    bg_color = "#0F172A"
    text_color = "#38BDF8"
    card_bg = "rgba(232, 121, 249, 0.1)"
    accent_color = "#F43F5E"

# Inject Custom CSS dynamically
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .main-card {{
        padding: 24px;
        border-radius: 12px;
        background-color: {card_bg};
        border: 1px solid {accent_color}33;
        margin-top: 20px;
        text-align: center;
    }}
    h1, h2, h3, label, .stMarkdown {{
        color: {text_color} !important;
    }}
    .stButton>button {{
        width: 100%;
        background-color: {accent_color} !important;
        color: white !important;
        border: none;
        padding: 10px 0px;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. MODEL LOADER
# ==========================================
@st.cache_resource
def load_model():
    with open('svc_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'svc_model.pkl' not found. Ensure it sits in the same directory as this script.")
    st.stop()


# ==========================================
# 3. INTERFACE & INPUTS
# ==========================================
st.title("🎓 Student Performance Prediction")
st.markdown("Adjust application styles in the sidebar. Fill out the student metrics below to evaluate classification via the SVC model.")
st.write("---")

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


# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
# Encode feature representations
gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet == "Yes" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0

edu_map = {"Uneducated/Primary": 0, "High School": 1, "Graduate": 2, "Post-Graduate": 3}
edu_encoded = edu_map[parent_edu]

# Array matches order: gender, age, study_hours_per_week, attendance_rate, parent_education, internet_access, extracurricular, previous_score, final_score
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

st.write("---")
if st.button("🔮 Run Analysis Prediction", type="primary"):
    with st.spinner("Calculating performance metrics..."):
        prediction = model.predict(features)[0]
        
        st.balloons()
        st.markdown(f"""
        <div class="main-card">
            <h3 style="margin: 0; padding-bottom: 8px;">Prediction Result</h3>
            <h1 style="color: {accent_color}; margin: 0; font-size: 2.5rem;">Class: {prediction}</h1>
        </div>
        """, unsafe_allow_html=True)
