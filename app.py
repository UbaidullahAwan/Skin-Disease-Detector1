import streamlit as st
import numpy as np
from PIL import Image
import pickle
import time
import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DermAI – Skin Disease Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  LOAD TRAINED MODEL
# ══════════════════════════════════════════════════════════════


@st.cache_resource
def load_model():
    """Load the trained skin disease classifier"""
    try:
        with open('skin_disease_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return clf, encoder
    except FileNotFoundError:
        return None, None


def extract_features_from_image(img):
    """Extract the same features used in training"""
    # Resize image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0

    # RGB channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]

    features = []

    # 1. Color features
    features.append(np.mean(r_channel))
    features.append(np.mean(g_channel))
    features.append(np.mean(b_channel))
    features.append(np.std(r_channel))
    features.append(np.std(g_channel))
    features.append(np.std(b_channel))

    # 2. Redness
    redness = np.mean(r_channel - g_channel)
    features.append(max(0, redness))

    # 3. Brightness
    brightness = (np.mean(r_channel) + np.mean(g_channel) +
                  np.mean(b_channel)) / 3
    features.append(brightness)

    # 4. Color variation
    features.append(np.std(img_array))

    # 5. Saturation
    max_rgb = np.maximum(np.maximum(r_channel, g_channel), b_channel)
    min_rgb = np.minimum(np.minimum(r_channel, g_channel), b_channel)
    saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 0.001))
    features.append(saturation)

    # 6. Texture features
    gray = np.mean(img_array, axis=2)
    h_edges = np.abs(np.diff(gray, axis=1))
    v_edges = np.abs(np.diff(gray, axis=0))

    if h_edges.size > 0:
        features.append(np.mean(h_edges))
    else:
        features.append(0)

    if v_edges.size > 0:
        features.append(np.mean(v_edges))
    else:
        features.append(0)

    total_edges = (np.mean(h_edges) if h_edges.size > 0 else 0) + \
                  (np.mean(v_edges) if v_edges.size > 0 else 0)
    features.append(total_edges / 2)

    # 7. Asymmetry
    h, w = gray.shape
    if h >= 2 and w >= 2:
        q1 = gray[:h//2, :w//2].mean()
        q2 = gray[:h//2, w//2:].mean()
        q3 = gray[h//2:, :w//2].mean()
        q4 = gray[h//2:, w//2:].mean()

        asymmetry_h = abs(q1 - q2) / (q1 + q2 + 0.001)
        asymmetry_v = abs(q3 - q4) / (q3 + q4 + 0.001)
        features.append(asymmetry_h)
        features.append(asymmetry_v)
        features.append((asymmetry_h + asymmetry_v) / 2)
    else:
        features.extend([0, 0, 0])

    # 8. Color histograms
    hist_r, _ = np.histogram(r_channel, bins=10, range=(0, 1))
    hist_g, _ = np.histogram(g_channel, bins=10, range=(0, 1))
    hist_b, _ = np.histogram(b_channel, bins=10, range=(0, 1))

    features.extend(hist_r / len(r_channel.flatten()))
    features.extend(hist_g / len(g_channel.flatten()))
    features.extend(hist_b / len(b_channel.flatten()))

    # 9. Texture uniformity
    features.append(np.std(gray))
    features.append(np.mean(gray) - np.std(gray))

    return np.array(features).reshape(1, -1)


def predict_skin_disease(img):
    """Predict skin disease using trained model"""
    clf, encoder = load_model()

    if clf is None:
        return None, "Model not trained. Run 'python3 train_model.py' first", 0

    # Extract features
    features = extract_features_from_image(img)

    # Get prediction and probabilities
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    confidence = np.max(probabilities)

    # Decode disease name
    disease_name = encoder.inverse_transform([prediction])[0]

    # Get probability distribution for all diseases
    all_probabilities = {}
    for i, disease in enumerate(encoder.classes_):
        all_probabilities[disease] = probabilities[i]

    return disease_name, confidence, all_probabilities


# ══════════════════════════════════════════════════════════════
#  THEMES
# ══════════════════════════════════════════════════════════════
THEMES = {
    "🌿 Skin Tone": {
        "bg": "linear-gradient(135deg,#f5e6d3 0%,#e8c9a0 30%,#d4a574 65%,#c8956c 100%)",
        "card": "rgba(255,247,237,0.93)",
        "border": "#d4956a",
        "primary": "#7a3b10",
        "accent": "#c07a3a",
        "text": "#2d1200",
        "subtext": "#7a4a2a",
        "sidebar": "linear-gradient(170deg,#eddcc8,#d5b08a)",
        "shadow": "rgba(122,59,16,0.16)",
        "hgrad": "linear-gradient(90deg,#7a3b10,#c07a3a,#9a5520)",
        "pill": "rgba(192,122,58,0.10)",
        "tag": "rgba(122,59,16,0.07)",
        "critical": "#b91c1c",
        "moderate": "#c2710a",
        "mild": "#166534",
    },
    "🌙 Midnight Dark": {
        "bg": "linear-gradient(135deg,#0d0b14 0%,#16112a 40%,#1e1535 100%)",
        "card": "rgba(22,17,42,0.96)",
        "border": "#5b2d8a",
        "primary": "#d8a4ff",
        "accent": "#a855f7",
        "text": "#f0e8ff",
        "subtext": "#c0a8e8",
        "sidebar": "linear-gradient(170deg,#100d1e,#1a1530)",
        "shadow": "rgba(168,85,247,0.25)",
        "hgrad": "linear-gradient(90deg,#d8a4ff,#e879f9,#818cf8)",
        "pill": "rgba(168,85,247,0.12)",
        "tag": "rgba(216,164,255,0.08)",
        "critical": "#f87171",
        "moderate": "#fb923c",
        "mild": "#4ade80",
    },
    "🏥 Clinical White": {
        "bg": "linear-gradient(135deg,#f0f7ff 0%,#ddeeff 50%,#c8e4fc 100%)",
        "card": "rgba(255,255,255,0.97)",
        "border": "#60a5fa",
        "primary": "#1e3a8a",
        "accent": "#2563eb",
        "text": "#0a1931",
        "subtext": "#2d4a7a",
        "sidebar": "linear-gradient(170deg,#e8f4ff,#c8e0fc)",
        "shadow": "rgba(37,99,235,0.12)",
        "hgrad": "linear-gradient(90deg,#1e3a8a,#2563eb,#0ea5e9)",
        "pill": "rgba(37,99,235,0.08)",
        "tag": "rgba(30,58,138,0.06)",
        "critical": "#dc2626",
        "moderate": "#d97706",
        "mild": "#16a34a",
    },
}

# ══════════════════════════════════════════════════════════════
#  DISEASE DATABASE
# ══════════════════════════════════════════════════════════════
DISEASE_DB = {
    "Acne Vulgaris": {
        "emoji": "🔵", "severity": "Mild", "urgency": "Routine",
        "specialist": "Dermatologist",
        "description": "A common skin condition causing pimples, blackheads, and whiteheads primarily on the face, chest, and back due to clogged hair follicles.",
        "medications": ["Benzoyl Peroxide 2.5–5% wash", "Adapalene (Differin) 0.1% Gel", "Salicylic Acid Cleanser 2%", "Clindamycin Topical 1%"],
        "recommendations": ["Wash face twice daily with gentle cleanser", "Never pop or squeeze pimples", "Use non-comedogenic sunscreen SPF 30+", "Change pillowcases twice a week", "See dermatologist if no improvement in 8 weeks"],
    },
    "Psoriasis": {
        "emoji": "🟡", "severity": "Moderate", "urgency": "Soon",
        "specialist": "Dermatologist / Rheumatologist",
        "description": "An autoimmune condition causing rapid skin cell buildup resulting in scaly silvery patches, redness, and inflammation — often cyclical with flare-ups.",
        "medications": ["Topical Corticosteroids (Betamethasone)", "Vitamin D Analogue – Calcipotriol cream", "Coal Tar Shampoo/Ointment", "Methotrexate (severe – oral, by prescription)"],
        "recommendations": ["Moisturize heavily with thick emollient creams", "Avoid triggers: stress, alcohol, smoking, infections", "Narrowband UVB phototherapy sessions", "Consult rheumatologist if joint pain present", "Avoid hot showers – use lukewarm water"],
    },
    "Melanoma": {
        "emoji": "🔴", "severity": "Critical", "urgency": "URGENT",
        "specialist": "Oncologist / Surgical Dermatologist",
        "description": "A dangerous form of skin cancer arising from melanocytes. Early detection is life-saving. Signs include asymmetry, irregular borders, multiple colors, diameter >6mm, or evolving lesions.",
        "medications": ["Pembrolizumab (Keytruda) – Immunotherapy", "Nivolumab (Opdivo) – PD-1 Inhibitor", "Dabrafenib + Trametinib – Targeted therapy", "Wide Local Excision – Surgical removal"],
        "recommendations": ["⚠️ SEEK IMMEDIATE MEDICAL ATTENTION TODAY", "Do not delay – same-week oncologist appointment", "Biopsy and staging required urgently", "Avoid all UV exposure – use SPF 50+ always", "Inform family – increased genetic risk for relatives"],
    },
    "Eczema (Atopic Dermatitis)": {
        "emoji": "🟠", "severity": "Mild–Moderate", "urgency": "Routine",
        "specialist": "Allergist / Dermatologist",
        "description": "A chronic inflammatory skin condition causing dry, intensely itchy, and inflamed patches. Triggered by allergens, stress, irritants, or temperature changes.",
        "medications": ["Hydrocortisone Cream 1% (OTC)", "Tacrolimus Ointment 0.1% (Protopic)", "Dupilumab (Dupixent) – for moderate-severe", "Cetirizine / Loratadine (antihistamine for itch)"],
        "recommendations": ["Moisturize within 3 mins of bathing – lock in moisture", "Use fragrance-free, hypoallergenic detergent", "Identify and strictly avoid personal triggers", "Cool compresses on flare-ups for relief", "Wear soft, breathable cotton clothing"],
    },
    "Ringworm (Tinea Corporis)": {
        "emoji": "🟢", "severity": "Mild", "urgency": "Routine",
        "specialist": "General Practitioner",
        "description": "A highly contagious fungal infection causing circular, ring-shaped, scaly, reddish patches with clearer skin in the center.",
        "medications": ["Clotrimazole Cream 1% (apply 2x daily × 4 weeks)", "Terbinafine (Lamisil) Cream 1%", "Miconazole Nitrate Cream 2%", "Oral Fluconazole 150mg (if extensive)"],
        "recommendations": ["Apply antifungal cream for full 4 weeks – don't stop early", "Keep area clean and completely dry", "Avoid sharing towels, clothing, or sports equipment", "Wash all bedding and towels in hot water", "Avoid tight-fitting synthetic clothing"],
    },
    "Rosacea": {
        "emoji": "🩷", "severity": "Mild–Moderate", "urgency": "Routine",
        "specialist": "Dermatologist",
        "description": "A chronic facial skin condition causing persistent redness, visible blood vessels (telangiectasia), and sometimes acne-like bumps. More common in fair-skinned individuals.",
        "medications": ["Metronidazole Gel/Cream 0.75% (topical)", "Azelaic Acid 15% Gel (Finacea)", "Brimonidine Gel 0.33% (for redness)", "Doxycycline 40mg delayed-release (oral, by Rx)"],
        "recommendations": ["Apply SPF 50+ mineral sunscreen daily", "Identify triggers: spicy foods, alcohol, hot drinks, sun", "Use only gentle, fragrance-free skincare products", "Consider vascular laser for persistent redness", "Avoid scrubbing or harsh exfoliants on face"],
    },
    "Basal Cell Carcinoma": {
        "emoji": "🔶", "severity": "Serious", "urgency": "Soon",
        "specialist": "Dermatologist / Oncologist",
        "description": "The most common skin cancer, usually appearing as a pearly or waxy bump, flat lesion, or bleeding sore on sun-exposed skin. Rarely spreads but requires removal.",
        "medications": ["Vismodegib (Erivedge) – for advanced/inoperable", "Imiquimod (Aldara) Cream 5% – superficial BCC", "5-Fluorouracil (Efudex) Cream – superficial", "Photodynamic Therapy (PDT) – clinic procedure"],
        "recommendations": ["Book dermatologist appointment within 2 weeks", "Mohs micrographic surgery is gold-standard treatment", "Annual full-body skin checks after diagnosis", "Strict sun avoidance – wide-brim hat + SPF 50+", "Do not squeeze or pick at lesion"],
    },
    "Vitiligo": {
        "emoji": "⚪", "severity": "Mild", "urgency": "Routine",
        "specialist": "Dermatologist",
        "description": "A condition where the immune system attacks melanocytes, causing loss of skin pigment in irregular white patches. Not contagious or life-threatening.",
        "medications": ["Ruxolitinib Cream 1.5% (Opzelura) – FDA approved", "Tacrolimus Ointment 0.1% – face/sensitive areas", "Topical Corticosteroids (short-term only)", "Afamelanotide implant + NB-UVB combination"],
        "recommendations": ["Apply SPF 50+ mineral sunscreen on all white patches", "Narrowband UVB phototherapy 3x/week", "Cosmetic camouflage for confidence", "Seek psychological support if impacting mental health", "Join vitiligo support community"],
    },

}

# ══════════════════════════════════════════════════════════════
#  CSS INJECTION
# ══════════════════════════════════════════════════════════════


def inject_css(t):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@300;400;500;600&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

html, body, [class*="css"] {{ font-family:'Outfit',sans-serif !important; }}

.stApp {{
    background:{t['bg']} !important;
    background-attachment:fixed !important;
}}

[data-testid="stSidebar"] {{
    background:{t['sidebar']} !important;
    border-right:1.5px solid {t['border']} !important;
}}
[data-testid="stSidebar"] * {{ color:{t['text']} !important; }}

.dcard {{
    background:{t['card']};
    border:1.5px solid {t['border']};
    border-radius:20px;
    padding:24px 26px;
    box-shadow:0 8px 36px {t['shadow']};
    backdrop-filter:blur(14px);
    margin-bottom:18px;
}}

.dtitle {{
    font-family:'Cormorant Garamond',serif;
    font-size:clamp(2rem,3.5vw,2.9rem);
    font-weight:700;
    background:{t['hgrad']};
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    line-height:1.2;
}}
.dsub {{
    color:{t['subtext']};
    font-size:.97rem;
    font-weight:300;
    margin-top:4px;
    letter-spacing:.02em;
    margin-bottom:24px;
}}

[data-testid="stFileUploader"] {{
    background:{t['card']} !important;
    border:2px dashed {t['border']} !important;
    border-radius:16px !important;
    padding:16px !important;
}}

.stButton>button {{
    background:{t['hgrad']} !important;
    color:#fff !important;
    border:none !important;
    border-radius:13px !important;
    font-size:1rem !important;
    font-weight:600 !important;
    font-family:'Outfit',sans-serif !important;
    letter-spacing:.03em !important;
    padding:12px 0 !important;
    box-shadow:0 4px 18px {t['shadow']} !important;
    transition:all .2s !important;
    width:100% !important;
}}
.stButton>button:hover {{ transform:translateY(-2px) !important; }}

.ctitle {{
    font-family:'Cormorant Garamond',serif;
    font-size:1.18rem; font-weight:600;
    color:{t['primary']};
    margin-bottom:14px;
    padding-bottom:9px;
    border-bottom:1.5px solid {t['border']};
}}
.ctitle i {{ margin-right: 8px; }}

.mpill {{ background:{t['pill']};border:1px solid {t['border']};border-radius:14px;padding:14px 10px;text-align:center; }}
.mval {{ font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:700;color:{t['primary']};line-height:1; }}
.mlbl {{ font-size:.68rem;color:{t['subtext']};text-transform:uppercase;letter-spacing:.07em;margin-top:4px; }}

.dname {{ font-family:'Cormorant Garamond',serif;font-size:1.7rem;font-weight:700;color:{t['primary']};line-height:1.2;margin-bottom:8px; }}
.ddesc {{ font-size:.87rem;color:{t['subtext']};line-height:1.75;margin:10px 0; }}

.meditem {{ background:{t['tag']};border-left:3px solid {t['accent']};padding:9px 13px;border-radius:0 11px 11px 0;margin-bottom:7px; }}
.recitem {{ background:{t['tag']};padding:9px 12px;border-radius:11px;margin-bottom:7px; }}

.crow {{ display:flex;align-items:center;gap:9px;margin-bottom:7px; }}
.cname {{ font-size:.76rem;color:{t['subtext']};width:155px; }}
.cbar {{ flex:1;height:7px;background:rgba(128,128,128,.15);border-radius:999px; }}
.cfill {{ height:100%;border-radius:999px;background:{t['hgrad']}; }}

.disclaim {{ background:rgba(185,28,28,.06);border:1px solid rgba(185,28,28,.25);border-radius:14px;padding:14px 18px;font-size:.79rem;margin-top:10px; }}
.pholder {{ text-align:center;padding:44px 0; }}
.sbl {{ font-size:.78rem;padding:3px 0;display:flex;gap:7px;align-items:center; }}
</style>""", unsafe_allow_html=True)


def render_results(disease_name, confidence, t):
    db = DISEASE_DB.get(disease_name, DISEASE_DB["Acne Vulgaris"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='mpill'><div class='mval'>{confidence*100:.1f}%</div><div class='mlbl'>Confidence</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='mpill'><div class='mval'>{db['emoji']}</div><div class='mlbl'>Severity</div></div>", unsafe_allow_html=True)
    with c3:
        icon = "🚨" if db['urgency'] == "URGENT" else (
            "⚡" if db['urgency'] == "Soon" else "✅")
        st.markdown(
            f"<div class='mpill'><div class='mval'>{icon}</div><div class='mlbl'>Urgency</div></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='dname'>{disease_name}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='ddesc'>{db['description']}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dspec'>🏥 Refer to: <b>{db['specialist']}</b></div>", unsafe_allow_html=True)


def render_meds(disease_name):
    db = DISEASE_DB.get(disease_name, DISEASE_DB["Acne Vulgaris"])
    for m in db["medications"]:
        st.markdown(
            f"<div class='meditem'>💊 {m}</div>", unsafe_allow_html=True)


def render_recs(disease_name):
    db = DISEASE_DB.get(disease_name, DISEASE_DB["Acne Vulgaris"])
    for r in db["recommendations"]:
        st.markdown(
            f"<div class='recitem'>→ {r}</div>", unsafe_allow_html=True)


def render_confidence(probs, t):
    for name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        emoji = DISEASE_DB.get(name, {}).get("emoji", "•")
        pct = int(prob * 100)
        st.markdown(
            f"<div class='crow'>"
            f"<span class='cname'>{emoji} {name[:20]}</span>"
            f"<div class='cbar'><div class='cfill' style='width:{pct}%'></div></div>"
            f"<span>{pct}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════


def render_sidebar(t):
    with st.sidebar:
        st.markdown(f"""
<div style='text-align:center;padding:20px 0'>
  <div style='font-size:3rem'>🔬</div>
  <div style='font-family:Cormorant Garamond,serif;font-size:1.5rem;font-weight:700;
              background:{t["hgrad"]};-webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    DermAI
  </div>
</div>""", unsafe_allow_html=True)

        theme_choice = st.selectbox(
            "🎨 Theme", list(THEMES.keys()), key="theme")

        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("1. Upload a skin image")
        st.markdown("2. Click Analyze")
        st.markdown("3. Get AI diagnosis")

        st.markdown("---")
        st.markdown("### 🦠 Detectable Diseases")
        for name in DISEASE_DB.keys():
            st.markdown(f"• {name}")

        st.markdown("---")
        st.markdown("*For educational purposes*")

        return theme_choice

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════


def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "🌿 Skin Tone"
    if "result" not in st.session_state:
        st.session_state.result = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "probs" not in st.session_state:
        st.session_state.probs = None

    t = THEMES[st.session_state.theme]
    inject_css(t)

    theme_choice = render_sidebar(t)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    # Header
    st.markdown("<div class='dtitle'>🔬 DermAI – Skin Disease Detector</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='dsub'>Upload a skin image · AI-powered diagnosis · Treatment recommendations</div>",
                unsafe_allow_html=True)

    # Check if model is loaded
    clf, encoder = load_model()
    if clf is None:
        st.warning(
            "⚠️ Model not found! Please run 'python3 train_model.py' first to train the AI.")
    else:
        st.success(
            f"✅ AI Model Loaded! Trained on {len(encoder.classes_)} diseases")

    # Layout
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='dcard'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ctitle'><i class='fas fa-camera'></i> Upload Image</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose a skin image", type=[
                                    "jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

            if st.button("🔍 Analyze Skin Condition", use_container_width=True):
                with st.spinner("🤖 AI analyzing..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)

                    disease, confidence, probs = predict_skin_disease(image)

                    if disease:
                        st.session_state.result = disease
                        st.session_state.confidence = confidence
                        st.session_state.probs = probs
                        st.rerun()
                    else:
                        st.error(
                            "Model not trained. Run 'python3 train_model.py' first")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='dcard'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ctitle'><i class='fas fa-stethoscope'></i> Diagnosis</div>", unsafe_allow_html=True)

        if st.session_state.result:
            render_results(st.session_state.result,
                           st.session_state.confidence, t)
        else:
            st.info("👈 Upload an image and click Analyze")

        st.markdown("</div>", unsafe_allow_html=True)

    # Results section
    if st.session_state.result:
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("<div class='dcard'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='ctitle'><i class='fas fa-pills'></i> Medications</div>", unsafe_allow_html=True)
            render_meds(st.session_state.result)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='dcard'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='ctitle'><i class='fas fa-heartbeat'></i> Self-Care</div>", unsafe_allow_html=True)
            render_recs(st.session_state.result)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c:
            st.markdown("<div class='dcard'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='ctitle'><i class='fas fa-chart-pie'></i> Probabilities</div>", unsafe_allow_html=True)
            render_confidence(st.session_state.probs, t)
            st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class='disclaim'>
    <i class='fas fa-exclamation-triangle'></i> <b>Medical Disclaimer:</b> For educational purposes only. 
    Always consult a doctor for medical advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
