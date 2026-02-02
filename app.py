import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Financial Forecasting System",
    layout="wide"
)

# =====================================================
# ADVANCED UI (ANIMATED + COLOR GRADED)
# =====================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 0% 0%, #1f2937, #020617 60%, #000000);
    color: #f9fafb;
    font-family: "Inter", system-ui, sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.2rem;
}

/* Animated Glass Card */
.glass-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,64,175,0.55));
    border-radius: 26px;
    padding: 1.8rem 2.2rem;
    border: 1px solid rgba(148,163,184,0.35);
    box-shadow:
        0 18px 45px rgba(0,0,0,0.65),
        0 0 0 1px rgba(148,163,184,0.15);
    backdrop-filter: blur(18px);
    transition: transform 220ms ease, box-shadow 220ms ease;
    margin-bottom: 1.8rem;
}

.glass-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow:
        0 32px 70px rgba(0,0,0,0.85),
        0 0 0 1px rgba(191,219,254,0.35);
}

/* Titles */
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(120deg, #e5e7eb, #93c5fd, #f97316);
    -webkit-background-clip: text;
    color: transparent;
}

.subtitle {
    color: #cbd5f5;
    font-size: 0.95rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(15,23,42,0.8));
    border-right: 1px solid rgba(148,163,184,0.45);
    backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Inputs */
.stNumberInput input, .stTextInput input {
    background: rgba(15,23,42,0.65) !important;
    border: 1px solid rgba(148,163,184,0.3) !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 999px;
    padding: 0.9rem;
    background: radial-gradient(circle at 0% 0%, #38bdf8, #2563eb, #4f46e5);
    color: white;
    font-weight: 700;
    border: none;
    box-shadow: 0 18px 30px rgba(37,99,235,0.45);
    transition: transform 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.8));
    border-radius: 18px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    color: #bfdbfe !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# =====================================================
# AUTH UTILS
# =====================================================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS finance_data (
    username TEXT,
    income REAL,
    total_expense REAL,
    predicted_savings REAL,
    savings_low REAL,
    savings_high REAL,
    expense_burden REAL,
    lifestyle_score REAL,
    optimization_score INTEGER,
    date TEXT
)
""")
conn.commit()

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    with open("models/linear_regression_savings.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

MODEL_FEATURES = [
    "Income","Rent","Loan_Repayment","Groceries","Transport",
    "Utilities","Healthcare","Education","Entertainment","Miscellaneous"
]

# =====================================================
# RECOMMENDATION ENGINE
# =====================================================
def generate_recommendations(exp, income):
    tips = []

    if income <= 0:
        return ["⚠️ Income is zero or invalid. Please enter a valid income."]

    # Core ratios
    rent_ratio = exp.get("Rent", 0) / income
    grocery_ratio = exp.get("Groceries", 0) / income
    entertainment_ratio = exp.get("Entertainment", 0) / income
    loan_ratio = exp.get("Loan", 0) / income
    utilities_ratio = exp.get("Utilities", 0) / income
    healthcare_ratio = exp.get("Healthcare", 0) / income
    education_ratio = exp.get("Education", 0) / income
    transport_ratio = exp.get("Transport", 0) / income

    total_expense = sum(exp.values())
    savings_ratio = (income - total_expense) / income

    # ---------------- HOUSING ----------------
    if rent_ratio > 0.35:
        tips.append("🏠 Rent exceeds 35% of income. Consider downsizing, sharing accommodation, or relocating.")
    elif rent_ratio < 0.20:
        tips.append("🏠 Housing costs are well managed. This supports long-term savings.")

    # ---------------- FOOD ----------------
    if grocery_ratio > 0.15:
        tips.append("🛒 Grocery spending is high. Monthly planning and bulk buying may reduce costs.")
    elif grocery_ratio < 0.08:
        tips.append("🛒 Grocery spending is efficient. Good budget control observed.")

    # ---------------- ENTERTAINMENT ----------------
    if entertainment_ratio > 0.10:
        tips.append("🎬 Entertainment expenses are high. Consider limiting discretionary outings.")
    elif entertainment_ratio > 0.05:
        tips.append("🎬 Entertainment spending is moderate. Monitor for unnecessary expenses.")

    # ---------------- LOANS ----------------
    if loan_ratio > 0.30:
        tips.append("💳 Loan repayments are heavy. Consider refinancing or prioritizing loan closure.")
    elif loan_ratio > 0.20:
        tips.append("💳 Loan burden is moderate. Avoid taking additional debt.")

    # ---------------- UTILITIES ----------------
    if utilities_ratio > 0.10:
        tips.append("⚡ Utility expenses are high. Energy-efficient usage may reduce bills.")

    # ---------------- TRANSPORT ----------------
    if transport_ratio > 0.12:
        tips.append("🚗 Transport expenses are high. Consider public transport or carpooling.")

    # ---------------- HEALTHCARE ----------------
    if healthcare_ratio > 0.10:
        tips.append("🏥 Healthcare expenses are high. Ensure adequate insurance coverage.")
    elif healthcare_ratio < 0.03:
        tips.append("🏥 Healthcare spending is low. Maintain regular health checkups.")

    # ---------------- EDUCATION ----------------
    if education_ratio > 0.15:
        tips.append("🎓 Education expenses are significant. Plan expenses with long-term ROI in mind.")

    # ---------------- SAVINGS HEALTH ----------------
    if savings_ratio < 0.10:
        tips.append("🚨 Savings rate is very low. Immediate expense optimization is recommended.")
    elif savings_ratio < 0.20:
        tips.append("⚠️ Savings rate is moderate. Increasing savings will improve financial security.")
    else:
        tips.append("✅ Savings rate is healthy. You are on track for long-term goals.")

    # ---------------- OVERALL FINANCIAL HEALTH ----------------
    if total_expense > income:
        tips.append("❗ Total expenses exceed income. This indicates financial stress and needs urgent correction.")

    # Fallback
    if not tips:
        tips.append("✅ Your spending pattern is well balanced. Keep up the good financial discipline.")

    return tips

# =====================================================
# LOGIN
# =====================================================
if not st.session_state.logged_in:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Smart Financial Forecasting</div>', unsafe_allow_html=True)

    action = st.radio("Action", ["Login","Sign Up"], horizontal=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button(action):
        if action == "Sign Up":
            cursor.execute("INSERT OR IGNORE INTO users VALUES (?,?)",(u,hash_password(p)))
            conn.commit()
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()
        else:
            cursor.execute("SELECT password FROM users WHERE username=?", (u,))
            row = cursor.fetchone()
            if row and row[0]==hash_password(p):
                st.session_state.logged_in=True
                st.session_state.username=u
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# MAIN DASHBOARD
# =====================================================
st.sidebar.success(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in=False
    st.session_state.username=""
    st.rerun()

st.markdown("""
<div class="glass-card">
  <div class="main-title">📊 Financial Intelligence Dashboard</div>
  <p class="subtitle">AI-powered savings prediction & smart spending insights</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT FORM
# =====================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
with st.form("finance_form"):
    income = st.number_input("Monthly Income", 0.0)

    c1,c2 = st.columns(2)
    with c1:
        rent = st.number_input("Rent",0.0)
        loan = st.number_input("Loan Repayment",0.0)
        groceries = st.number_input("Groceries",0.0)
        transport = st.number_input("Transport",0.0)
        utilities = st.number_input("Utilities",0.0)
    with c2:
        healthcare = st.number_input("Healthcare",0.0)
        education = st.number_input("Education",0.0)
        entertainment = st.number_input("Entertainment",0.0)
        misc = st.number_input("Miscellaneous",0.0)

    submit = st.form_submit_button("Analyze")
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROCESS
# =====================================================
if submit and income>0:
    expenses = {
        "Rent": rent,
        "Loan": loan,
        "Groceries": groceries,
        "Transport": transport,
        "Utilities": utilities,
        "Healthcare": healthcare,
        "Education": education,
        "Entertainment": entertainment,
        "Miscellaneous": misc
    }

    total_exp = sum(expenses.values())
    X = pd.DataFrame([{
        "Income": income,
        "Rent": rent,
        "Loan_Repayment": loan,
        "Groceries": groceries,
        "Transport": transport,
        "Utilities": utilities,
        "Healthcare": healthcare,
        "Education": education,
        "Entertainment": entertainment,
        "Miscellaneous": misc
    }])[MODEL_FEATURES]

    predicted_savings = max(0, model.predict(X)[0])

    # ================= SUMMARY =================
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    m1.metric("Income", f"₹{income:,.0f}")
    m2.metric("Expenses", f"₹{total_exp:,.0f}")
    m3.metric("Predicted Savings", f"₹{predicted_savings:,.0f}")

    # ================= PIE CHART =================
    st.subheader("🧩 Expense Distribution")

    COLORS = [
        "#60a5fa","#f87171","#34d399","#fbbf24",
        "#a78bfa","#fb7185","#22d3ee","#f97316","#94a3b8"
    ]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(expenses.values(), labels=expenses.keys(), autopct="%1.1f%%",
           startangle=140, colors=COLORS, wedgeprops={"edgecolor":"white"})
    ax.axis("equal")
    st.pyplot(fig)

    # ================= RECOMMENDATIONS =================
    st.subheader("🤖 Smart Recommendations")
    for tip in generate_recommendations(expenses, income):
        st.markdown(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Smart Financial Forecasting System • FinTech ML Dashboard")
