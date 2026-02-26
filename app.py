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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #0b0f1a;
    --surface: #111827;
    --surface2: #1a2236;
    --accent: #3b82f6;
    --accent2: #06b6d4;
    --green: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: rgba(59,130,246,0.15);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stSelectbox > div, .stRadio > div {
    background: transparent !important;
    color: var(--text) !important;
}

[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(59,130,246,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(6,182,212,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.login-box {
    max-width: 420px;
    margin: 6rem auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
}
.login-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #60a5fa;
    margin-bottom: 1.5rem;
}

stSlider > div { color: var(--text) !important; }
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
# INR FORMAT / PARSE UTILS
# =====================================================
def format_inr(x):
    return f"₹{x:,.0f}"

def parse_amount(value):
    value = str(value).replace("₹", "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0

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
        username TEXT PRIMARY KEY,
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
        return ["Income is zero or invalid. Please enter a valid income."]

    rent_ratio         = exp.get("Rent", 0) / income
    grocery_ratio      = exp.get("Groceries", 0) / income
    entertainment_ratio= exp.get("Entertainment", 0) / income
    loan_ratio         = exp.get("Loan", 0) / income
    utilities_ratio    = exp.get("Utilities", 0) / income
    healthcare_ratio   = exp.get("Healthcare", 0) / income
    education_ratio    = exp.get("Education", 0) / income
    transport_ratio    = exp.get("Transport", 0) / income
    total_expense      = sum(exp.values())
    savings_ratio      = (income - total_expense) / income

    if rent_ratio > 0.35:
        tips.append("Rent exceeds 35% of income. Consider downsizing, sharing accommodation, or relocating.")
    elif rent_ratio < 0.20:
        tips.append("Housing costs are well managed. This supports long-term savings.")

    if grocery_ratio > 0.15:
        tips.append("Grocery spending is high. Monthly planning and bulk buying may reduce costs.")
    elif grocery_ratio < 0.08:
        tips.append("Grocery spending is efficient. Good budget control observed.")

    if entertainment_ratio > 0.10:
        tips.append("Entertainment expenses are high. Consider limiting discretionary outings.")
    elif entertainment_ratio > 0.05:
        tips.append("Entertainment spending is moderate. Monitor for unnecessary expenses.")

    if loan_ratio > 0.30:
        tips.append("Loan repayments are heavy. Consider refinancing or prioritizing loan closure.")
    elif loan_ratio > 0.20:
        tips.append("Loan burden is moderate. Avoid taking additional debt.")

    if utilities_ratio > 0.10:
        tips.append("Utility expenses are high. Energy-efficient usage may reduce bills.")

    if transport_ratio > 0.12:
        tips.append("Transport expenses are high. Consider public transport or carpooling.")

    if healthcare_ratio > 0.10:
        tips.append("Healthcare expenses are high. Ensure adequate insurance coverage.")
    elif healthcare_ratio < 0.03:
        tips.append("Healthcare spending is low. Maintain regular health checkups.")

    if education_ratio > 0.15:
        tips.append("Education expenses are significant. Plan expenses with long-term ROI in mind.")

    if savings_ratio < 0.10:
        tips.append("Savings rate is very low. Immediate expense optimization is recommended.")
    elif savings_ratio < 0.20:
        tips.append("Savings rate is moderate. Increasing savings will improve financial security.")
    else:
        tips.append("Savings rate is healthy. You are on track for long-term goals.")

    if total_expense > income:
        tips.append("Total expenses exceed income. This indicates financial stress and needs urgent correction.")

    if not tips:
        tips.append("Your spending pattern is well balanced. Keep up the good financial discipline.")

    return tips

# =====================================================
# LOGIN
# =====================================================
if not st.session_state.logged_in:
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Smart Financial Forecasting</div>', unsafe_allow_html=True)

    action = st.radio("Action", ["Login", "Sign Up"], horizontal=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button(action):
        if action == "Sign Up":
            cursor.execute("SELECT username FROM users WHERE username=?", (u,))
            existing_user = cursor.fetchone()
            if existing_user:
                st.error("Username already exists. Please choose another one.")
            else:
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (u, hash_password(p))
                )
                conn.commit()
                st.success("Account created successfully! Please login.")
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
        else:
            cursor.execute("SELECT password FROM users WHERE username=?", (u,))
            row = cursor.fetchone()
            if row and row[0] == hash_password(p):
                st.session_state.logged_in = True
                st.session_state.username = u
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
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Financial Intelligence Dashboard</div>
    <div class="hero-sub">AI-powered savings prediction &amp; smart spending insights</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT MODE TOGGLE
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

mode = st.radio("Input Mode", ["Amount ₹", "Percentage %"], horizontal=True)

# Income input (text-based for ₹ formatting)
income_input = st.text_input("Monthly Income", "₹0")
income = parse_amount(income_input)

st.markdown("### Monthly Expenses")

CATEGORIES = [
    "Rent", "Loan Repayment", "Groceries", "Transport", "Utilities",
    "Healthcare", "Education", "Entertainment", "Miscellaneous"
]

values = {}

# =====================================================
# AMOUNT MODE
# =====================================================
if mode == "Amount ₹":
    cols = st.columns(2)
    for i, cat in enumerate(CATEGORIES):
        with cols[i % 2]:
            val = st.text_input(cat, "₹0", key=f"amt_{cat}")
            values[cat] = parse_amount(val)

# =====================================================
# PERCENTAGE MODE
# =====================================================
else:
    for cat in CATEGORIES:
        percent = st.slider(cat, 0, 100, 10, key=f"pct_{cat}")
        values[cat] = (percent / 100) * income

analyze_btn = st.button("Analyze")
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROCESS
# =====================================================
if analyze_btn and income > 0:
    # Map category names to model keys
    rent        = values["Rent"]
    loan        = values["Loan Repayment"]
    groceries   = values["Groceries"]
    transport   = values["Transport"]
    utilities   = values["Utilities"]
    healthcare  = values["Healthcare"]
    education   = values["Education"]
    entertainment = values["Entertainment"]
    misc        = values["Miscellaneous"]

    total_exp = sum(values.values())

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

    # ================= SUMMARY METRICS =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Income",            format_inr(income))
    m2.metric("Expenses",          format_inr(total_exp))
    m3.metric("Predicted Savings", format_inr(predicted_savings))
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= PIE CHART =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Expense Distribution")

    COLORS = [
        "#60a5fa","#f87171","#34d399","#fbbf24",
        "#a78bfa","#fb7185","#22d3ee","#f97316","#94a3b8"
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.pie(
        values.values(),
        labels=values.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=COLORS,
        wedgeprops={"edgecolor": "#1a2236", "linewidth": 1.5},
        textprops={"color": "#e2e8f0", "fontsize": 10}
    )
    ax.axis("equal")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= RECOMMENDATIONS =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Smart Recommendations")

    # Build expense dict with key names matching recommendation engine
    exp_for_reco = {
        "Rent":          rent,
        "Loan":          loan,
        "Groceries":     groceries,
        "Transport":     transport,
        "Utilities":     utilities,
        "Healthcare":    healthcare,
        "Education":     education,
        "Entertainment": entertainment,
        "Miscellaneous": misc,
    }

    for tip in generate_recommendations(exp_for_reco, income):
        st.markdown(f"- {tip}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= SAVE TO DB =================
    cursor.execute("""
        INSERT OR REPLACE INTO finance_data
        (username, income, total_expense, predicted_savings, date)
        VALUES (?, ?, ?, ?, ?)
    """, (
        st.session_state.username,
        income,
        total_exp,
        predicted_savings,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

elif analyze_btn and income <= 0:
    st.warning("Please enter a valid income greater than 0.")

st.caption("Smart Financial Forecasting System • FinTech ML Dashboard")