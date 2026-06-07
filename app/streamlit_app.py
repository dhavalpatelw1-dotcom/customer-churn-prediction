# from pathlib import Path
# import streamlit as st
# import joblib
# import pandas as pd

# BASE_DIR = Path(__file__).resolve().parent.parent
# #st.write(BASE_DIR)
# model = joblib.load(
#     BASE_DIR / "models" / "churn_model.pkl"
# )

# scaler = joblib.load(
#     BASE_DIR / "models" / "scaler.pkl"
# )

# feature_names = joblib.load(
#     BASE_DIR / "models" / "feature_names.pkl"
# )

# st.title("Customer Churn Prediction")

# tenure = st.number_input(
#     "Tenure (months)",
#     min_value=0,
#     value=12
# )

# monthly_charge = st.number_input(
#     "Monthly Charges",
#     min_value=0.0,
#     value=75.0
# )

# total_charge = st.number_input(
#     "Total Charges",
#     min_value=0.0,
#     value=900.0
# )

# if st.button("Predict"):

#     row = [tenure, monthly_charge, total_charge]

#     remaining = len(feature_names) - 3

#     row.extend([0] * remaining)

#     sample = pd.DataFrame(
#         [row],
#         columns=feature_names
#     )

#     sample_scaled = scaler.transform(
#         sample
#     )

#     prediction = model.predict(
#         sample_scaled
#     )[0]

#     if prediction == 1:
#         st.error(
#             "Customer likely to churn"
#         )
#     else:
#         st.success(
#             "Customer likely to stay"
#         )


## version 2.0
# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("models/churn_model.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# st.title("Customer Churn Prediction System")

# tenure = st.number_input("Tenure", 0, 100, 1)
# monthly = st.number_input("Monthly Charges", 0, 200, 70)
# total = st.number_input("Total Charges", 0, 10000, 1000)

# contract = st.selectbox(
#     "Contract",
#     ["Month-to-month", "One year", "Two year"]
# )

# internet = st.selectbox(
#     "Internet Service",
#     ["DSL", "Fiber optic", "No"]
# )

# if st.button("Predict"):

#     data = {
#         "tenure": tenure,
#         "MonthlyCharges": monthly,
#         "TotalCharges": total,
#         "Contract": contract,
#         "InternetService": internet
#     }

#     df = pd.DataFrame([data])

#     df = pd.get_dummies(df)

#     df = df.reindex(columns=feature_names, fill_value=0)

#     df_scaled = scaler.transform(df)

#     pred = model.predict(df_scaled)[0]
#     prob = model.predict_proba(df_scaled)[0]

#     st.subheader("Result")

#     if pred == 1:
#         st.error("Customer will CHURN")
#     else:
#         st.success("Customer will STAY")

#     st.write("Churn Probability:", round(prob[1], 2))



####Version 3.0

import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield · Prediction",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #08090d;
    color: #e8e9f0;
}

.stApp {
    background: #08090d;
}

/* ── Noise grain overlay ── */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7DF9A0;
    background: rgba(125,249,160,0.08);
    border: 1px solid rgba(125,249,160,0.25);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 30%, #7DF9A0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
}

.hero-sub {
    color: #7a7f94;
    font-size: 1rem;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(125,249,160,0.4), transparent);
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7DF9A0;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(125,249,160,0.15);
}

/* ── Input overrides ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e9f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: rgba(125,249,160,0.5) !important;
    box-shadow: 0 0 0 3px rgba(125,249,160,0.08) !important;
}

label, .stNumberInput label, .stSelectbox label {
    color: #9da3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7DF9A0 0%, #34d058 100%) !important;
    color: #0a1a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(125,249,160,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(125,249,160,0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-churn {
    background: rgba(255, 59, 48, 0.08);
    border: 1px solid rgba(255, 59, 48, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}

.result-stay {
    background: rgba(125, 249, 160, 0.08);
    border: 1px solid rgba(125, 249, 160, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}

.result-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
}

.result-title-churn { color: #ff6b6b; }
.result-title-stay  { color: #7DF9A0; }

.result-sub {
    color: #7a7f94;
    font-size: 0.9rem;
}

/* ── Probability meter ── */
.prob-bar-wrap {
    margin-top: 1.4rem;
}

.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.prob-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a7f94;
}

.prob-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
}

.prob-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #3a3f52;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 680px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model        = joblib.load("models/churn_model.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_models()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ &nbsp; AI-Powered Analytics</div>
    <h1 class="hero-title">ChurnShield</h1>
    <p class="hero-sub"><marquee>Predict customer churn before it happens — powered by ML.</marquee></p>
</div>
""", unsafe_allow_html=True)


# ── Usage Metrics Card ─────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="section-label">📊 &nbsp; Usage Metrics</div>
    <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.04); border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#7a7f94; margin-bottom:0.3rem;">Monthly Charges</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e9f0;">$<span id="mc_display">70</span></div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.04); border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#7a7f94; margin-bottom:0.3rem;">Tenure</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e9f0;"><span id="tenure_display">1</span> <span style="font-size:1rem; color:#7a7f94;">mo</span></div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.04); border-radius:12px; padding:1rem 1.2rem;">
            <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#7a7f94; margin-bottom:0.3rem;">Total Charges</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e9f0;">$<span id="tc_display">1000</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input Card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="section-label">👤 &nbsp; Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure  = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly = st.number_input("Monthly Charges ($)", min_value=0, max_value=200, value=70)
with col2:
    total    = st.number_input("Total Charges ($)", min_value=0, max_value=10000, value=1000)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
predict = st.button("⚡  Run Churn Prediction")


# ── Prediction Logic & Result Card ───────────────────────────────────────────
if predict:
    data = {
        "tenure":          tenure,
        "MonthlyCharges":  monthly,
        "TotalCharges":    total,
        "Contract":        contract,
        "InternetService": internet,
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]

    churn_prob = round(prob[1] * 100, 1)
    stay_prob  = round(prob[0] * 100, 1)

    # Result card
    if pred == 1:
        bar_color = "#ff6b6b"
        st.markdown(f"""
        <div class="result-churn">
            <div class="result-icon">⚠️</div>
            <div class="result-title result-title-churn">High Churn Risk</div>
            <div class="result-sub">This customer is likely to leave. Consider retention actions.</div>
            <div class="prob-bar-wrap">
                <div class="prob-label-row">
                    <span class="prob-label">Churn Probability</span>
                    <span class="prob-value" style="color:#ff6b6b;">{churn_prob}%</span>
                </div>
                <div class="prob-track">
                    <div style="width:{churn_prob}%; background:linear-gradient(90deg,#ff6b6b,#ff3b30);
                                height:8px; border-radius:999px; transition:width 0.6s ease;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bar_color = "#7DF9A0"
        st.markdown(f"""
        <div class="result-stay">
            <div class="result-icon">✅</div>
            <div class="result-title result-title-stay">Customer Will Stay</div>
            <div class="result-sub">Low churn risk detected. Customer appears satisfied.</div>
            <div class="prob-bar-wrap">
                <div class="prob-label-row">
                    <span class="prob-label">Retention Probability</span>
                    <span class="prob-value" style="color:#7DF9A0;">{stay_prob}%</span>
                </div>
                <div class="prob-track">
                    <div style="width:{stay_prob}%; background:linear-gradient(90deg,#7DF9A0,#34d058);
                                height:8px; border-radius:999px; transition:width 0.6s ease;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Breakdown row
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="section-label">📈 &nbsp; Probability Breakdown</div>
        <div style="display:flex; gap:1rem;">
            <div style="flex:1; text-align:center; background:rgba(255,107,107,0.06);
                        border:1px solid rgba(255,107,107,0.2); border-radius:12px; padding:1rem;">
                <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                            color:#7a7f94; margin-bottom:0.3rem;">Will Churn</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                            color:#ff6b6b;">{churn_prob}<span style="font-size:1rem">%</span></div>
            </div>
            <div style="flex:1; text-align:center; background:rgba(125,249,160,0.06);
                        border:1px solid rgba(125,249,160,0.2); border-radius:12px; padding:1rem;">
                <div style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                            color:#7a7f94; margin-bottom:0.3rem;">Will Stay</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                            color:#7DF9A0;">{stay_prob}<span style="font-size:1rem">%</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">ChurnShield · Powered by ML · Built with Streamlit</div>', unsafe_allow_html=True)