import streamlit as st
import numpy as np
from PIL import Image
import random

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Skin Disease AI Detector",
                   page_icon="🧴", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
/* ---- Base ---- */
.stApp {
    background-color: #fdf6ec;
}

/* ---- Header ---- */
.app-header {
    background: linear-gradient(90deg, #c97b5a 0%, #e8a87c 60%, #f5cc8a 100%);
    padding: 1.4rem 2rem 1.2rem;
    border-radius: 0 0 16px 16px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
    border-bottom: 2px solid #d4945e;
}
.header-icon {
    width: 46px;
    height: 46px;
    background: rgba(255,255,255,0.25);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.header-title {
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: 0.01em;
}
.header-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.85);
    margin: 2px 0 0;
}

/* ---- Panels ---- */
.panel {
    background: #fffdf7;
    border: 0.5px solid #e8c8a4;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 11px;
    font-weight: 600;
    color: #9a5f3a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 7px;
}
.panel-title::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #c97b5a;
    flex-shrink: 0;
}

/* ---- Prediction card ---- */
.prediction-card {
    background: #fff8f0;
    border: 0.5px solid #e8c8a4;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.pred-label {
    font-size: 11px;
    color: #b07850;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.pred-value {
    font-size: 22px;
    font-weight: 700;
    color: #6b3d22;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
}
.conf-bar-wrap {
    flex: 1;
    background: #eed8bc;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.conf-bar {
    height: 6px;
    background: linear-gradient(90deg, #c97b5a, #e8a87c);
    border-radius: 4px;
}
.conf-pct {
    font-size: 13px;
    color: #9a5f3a;
    font-weight: 600;
    min-width: 38px;
    text-align: right;
}

/* ---- Confidence badges ---- */
.badge {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.badge-high   { background: #e8f8ed; color: #2e7d52; }
.badge-medium { background: #fff4e0; color: #9a6010; }
.badge-low    { background: #fdeaea; color: #c0392b; }

/* ---- Recommendation card ---- */
.rec-card {
    background: #fffce8;
    border: 0.5px solid #e8d87a;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}
.rec-label {
    font-size: 11px;
    color: #9a8a30;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.rec-text {
    font-size: 14px;
    color: #5a5020;
    line-height: 1.65;
}

/* ---- Metric mini cards ---- */
.metric-row {
    display: flex;
    gap: 10px;
    margin-top: 0.2rem;
}
.metric-card {
    flex: 1;
    background: #f5ede0;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.metric-label {
    font-size: 10px;
    color: #b07850;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-size: 15px;
    font-weight: 700;
    color: #6b3d22;
    margin-top: 3px;
}

/* ---- Tips card ---- */
.tips-card {
    background: #f5ede0;
    border-radius: 10px;
    border: 0.5px solid #d4b090;
    padding: 0.85rem 1rem;
    margin-top: 0.85rem;
}
.tips-title {
    font-size: 11px;
    color: #9a5f3a;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.tips-body {
    font-size: 12px;
    color: #7a5040;
    line-height: 1.8;
}

/* ---- Disclaimer ---- */
.disclaimer {
    background: #fff9f0;
    border: 0.5px solid #e8c8a4;
    border-left: 3px solid #e8a060;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.disclaimer-text {
    font-size: 12px;
    color: #a07050;
    line-height: 1.55;
}

/* ---- Info box ---- */
.info-box {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #fffce8;
    border: 0.5px solid #e8d87a;
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.info-text {
    font-size: 13px;
    color: #7a6a20;
    line-height: 1.6;
}

/* ---- Streamlit overrides ---- */
[data-testid="stFileUploader"] {
    background: #fff9f2;
    border: 1.5px dashed #d4a070 !important;
    border-radius: 10px !important;
    padding: 0.5rem;
}
div[data-testid="column"] > div {
    height: 100%;
}
h3 {
    color: #9a5f3a !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("""
<div class="app-header">
    <div class="header-icon">🧴</div>
    <div>
        <div class="header-title">Skin Disease AI Detector</div>
        <div class="header-sub">Upload a skin image for instant AI-based analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= DATA =================
classes = [
    "Acne", "Eczema", "Psoriasis", "Fungal Infection",
    "Rosacea", "Herpes", "Impetigo", "Melanoma",
    "Dermatitis", "Vitiligo"
]

recommendations = {
    "Acne": "Cleanse skin gently twice daily with a non-comedogenic cleanser. Avoid touching your face and reduce intake of sugary or oily foods.",
    "Eczema": "Apply a fragrance-free moisturizer regularly. Avoid harsh soaps, synthetic fabrics, and known skin irritants.",
    "Psoriasis": "Keep skin hydrated and follow prescribed topical treatments. Avoid stress triggers and consult a dermatologist for systemic options.",
    "Fungal Infection": "Keep the affected area clean and dry. Apply antifungal cream as directed and avoid sharing personal items.",
    "Rosacea": "Protect skin from sun exposure with SPF 30+. Avoid spicy foods, alcohol, and extreme temperatures.",
    "Herpes": "Avoid direct skin contact during outbreaks. Maintain strict hygiene and consult a doctor for antiviral therapy.",
    "Impetigo": "Keep the area clean and avoid scratching. Wash hands frequently and seek antibiotic treatment if widespread.",
    "Melanoma": "Seek immediate evaluation from a dermatologist or oncologist. Early detection is critical — do not delay medical attention.",
    "Dermatitis": "Identify and avoid allergens or irritants. Use gentle, fragrance-free products and a soothing barrier cream.",
    "Vitiligo": "Protect depigmented areas from sun exposure. Consult a dermatologist for phototherapy or topical treatment options.",
}

urgency_map = {
    "Acne": ("Low", "Dermatologist"),
    "Eczema": ("Moderate", "Dermatologist"),
    "Psoriasis": ("Moderate", "Dermatologist"),
    "Fungal Infection": ("Moderate", "GP / Dermatologist"),
    "Rosacea": ("Low", "Dermatologist"),
    "Herpes": ("High", "GP / Specialist"),
    "Impetigo": ("Moderate", "GP"),
    "Melanoma": ("Urgent", "Oncologist"),
    "Dermatitis": ("Low", "Dermatologist"),
    "Vitiligo": ("Low", "Dermatologist"),
}


# ================= SIMULATION =================
def predict_image():
    prediction = random.choice(classes)
    confidence = random.randint(70, 95)
    return prediction, confidence


def get_conf_meta(conf):
    if conf > 85:
        return "badge-high", "High confidence"
    elif conf > 75:
        return "badge-medium", "Medium confidence"
    else:
        return "badge-low", "Low confidence"


# ================= LAYOUT =================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="panel-title">Upload image</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse — JPG, JPEG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded image")

    st.markdown("""
    <div class="tips-card">
        <div class="tips-title">Tips for better results</div>
        <div class="tips-body">
            · Use good natural lighting<br>
            · Capture the affected area clearly<br>
            · Avoid blurry or distant shots<br>
            · A plain background helps accuracy<br>
            · Accepted formats: JPG, JPEG, PNG, WEBP
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-title">Analysis results</div>',
                unsafe_allow_html=True)

    if uploaded_file:
        prediction, confidence = predict_image()
        badge_cls, badge_label = get_conf_meta(confidence)
        urgency, specialist = urgency_map[prediction]
        bar_width = confidence

        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-label">Detected condition</div>
            <div class="pred-value">{prediction}</div>
            <div class="conf-row">
                <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{bar_width}%;"></div>
                </div>
                <div class="conf-pct">{confidence}%</div>
            </div>
            <span class="badge {badge_cls}">{badge_label}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-label">Recommendation</div>
            <div class="rec-text">{recommendations[prediction]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Urgency</div>
                <div class="metric-value">{urgency}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">See a</div>
                <div class="metric-value">{specialist}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
            <div class="info-text">
                Upload a clear image of the affected skin area on the left. The AI will detect the most likely
                condition and provide care recommendations instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================= DISCLAIMER =================
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-text">
        ⚠ This tool is intended for informational purposes only and does not constitute medical advice.
        Always consult a qualified dermatologist or healthcare professional for an accurate diagnosis and treatment plan.
    </div>
</div>
""", unsafe_allow_html=True)
