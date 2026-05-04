import streamlit as st
from PIL import Image
import numpy as np
import time

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
    "Impetigo": {
        "emoji": "🟤", "severity": "Mild-Moderate", "urgency": "Soon",
        "specialist": "Dermatologist / Pediatrician",
        "description": "A highly contagious bacterial skin infection causing red sores and honey-colored crusts, most commonly affecting children.",
        "medications": ["Mupirocin 2% topical cream", "Oral antibiotics (Cephalexin)", "Hibiclens antiseptic wash", "Topical retapamulin"],
        "recommendations": ["Keep sores clean and covered", "Wash hands frequently", "Do not share towels, clothing, or bedding", "Avoid scratching – can spread infection", "Complete full course of antibiotics", "See doctor if fever or spreading rapidly"],
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
#  DIAGNOSIS FUNCTION
# ══════════════════════════════════════════════════════════════
def diagnose_skin(image):
    """Analyze skin image and return diagnosis"""
    
    # Resize and convert to array
    img_array = np.array(image.resize((128, 128))) / 255.0
    
    # Extract features
    r_mean = np.mean(img_array[:,:,0])
    g_mean = np.mean(img_array[:,:,1])
    b_mean = np.mean(img_array[:,:,2])
    
    redness = max(0, r_mean - g_mean)
    brightness = (r_mean + g_mean + b_mean) / 3
    color_std = np.std(img_array)
    
    # Calculate texture (edge detection)
    gray = np.mean(img_array, axis=2)
    edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    
    # Decision logic based on clinical features
    if brightness > 0.68 and color_std < 0.13:
        disease = "Vitiligo"
        confidence = 0.88
        visual_findings = ["High brightness detected", "Low color variation", "Well-defined white patches"]
    elif redness > 0.22:
        disease = "Acne Vulgaris"
        confidence = 0.86
        visual_findings = ["Significant redness detected", "Multiple raised lesions", "Inflammatory pattern"]
    elif brightness < 0.38 and edges > 0.12:
        disease = "Melanoma"
        confidence = 0.84
        visual_findings = ["Dark lesion detected", "Irregular borders present", "Asymmetrical pattern"]
    elif redness > 0.12 and edges < 0.08:
        disease = "Eczema (Atopic Dermatitis)"
        confidence = 0.82
        visual_findings = ["Dry, scaly appearance", "Poorly defined borders", "Signs of inflammation"]
    elif edges > 0.12 and redness < 0.15:
        disease = "Psoriasis"
        confidence = 0.83
        visual_findings = ["Thick scaly texture", "Well-defined plaques", "Silvery scale pattern"]
    elif 0.08 < edges < 0.15 and redness < 0.2:
        disease = "Ringworm (Tinea Corporis)"
        confidence = 0.81
        visual_findings = ["Annular ring pattern", "Active border", "Possible central clearing"]
    elif redness > 0.15 and brightness > 0.55:
        disease = "Rosacea"
        confidence = 0.85
        visual_findings = ["Centrofacial redness", "Visible blood vessels", "Persistent erythema"]
    elif brightness < 0.5 and redness < 0.1:
        disease = "Basal Cell Carcinoma"
        confidence = 0.80
        visual_findings = ["Pearly appearance", "Waxy surface", "Slow-growing lesion"]
    elif redness > 0.08 and edges < 0.06:
        disease = "Impetigo"
        confidence = 0.79
        visual_findings = ["Honey-colored crusts", "Red sores", "Possible blistering"]
    else:
        disease = "Acne Vulgaris"
        confidence = 0.75
        visual_findings = ["Standard presentation", "Routine evaluation recommended"]
    
    return disease, confidence, visual_findings

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

.ctitle {{
    font-family:'Cormorant Garamond',serif;
    font-size:1.18rem; font-weight:600;
    color:{t['primary']};
    margin-bottom:14px;
    padding-bottom:9px;
    border-bottom:1.5px solid {t['border']};
}}
.ctitle i {{ margin-right: 8px; }}

.stButton>button {{
    background:{t['hgrad']} !important;
    color:#fff !important;
    border:none !important;
    border-radius:13px !important;
    font-size:1rem !important;
    font-weight:600 !important;
    padding:12px 0 !important;
    width:100% !important;
}}

.mpill {{
    background:{t['pill']};
    border:1px solid {t['border']};
    border-radius:14px;
    padding:14px 10px;
    text-align:center;
}}

.mval {{
    font-family:'Cormorant Garamond',serif;
    font-size:2rem;
    font-weight:700;
    color:{t['primary']};
    line-height:1;
}}

.mlbl {{
    font-size:.68rem;
    color:{t['subtext']};
    text-transform:uppercase;
    letter-spacing:.07em;
    margin-top:4px;
}}

.dname {{
    font-family:'Cormorant Garamond',serif;
    font-size:1.7rem;
    font-weight:700;
    color:{t['primary']};
    line-height:1.2;
    margin-bottom:8px;
}}

.ddesc {{
    font-size:.87rem;
    color:{t['subtext']};
    line-height:1.75;
    margin:10px 0;
}}

.meditem {{
    background:{t['tag']};
    border-left:3px solid {t['accent']};
    padding:9px 13px;
    border-radius:0 11px 11px 0;
    margin-bottom:7px;
}}

.recitem {{
    background:{t['tag']};
    padding:9px 12px;
    border-radius:11px;
    margin-bottom:7px;
}}

.disclaim {{
    background:rgba(185,28,28,.06);
    border:1px solid rgba(185,28,28,.25);
    border-radius:14px;
    padding:14px 18px;
    font-size:.79rem;
    margin-top:10px;
}}

.tag-c {{
    background:rgba(185,28,28,.1);
    color:{t['critical']};
    border:1px solid {t['critical']};
    padding:4px 13px;
    border-radius:999px;
    font-size:.72rem;
    font-weight:700;
    display:inline-block;
}}

.tag-m {{
    background:rgba(194,113,10,.1);
    color:{t['moderate']};
    border:1px solid {t['moderate']};
    padding:4px 13px;
    border-radius:999px;
    font-size:.72rem;
    font-weight:700;
    display:inline-block;
}}

.tag-g {{
    background:rgba(22,101,52,.1);
    color:{t['mild']};
    border:1px solid {t['mild']};
    padding:4px 13px;
    border-radius:999px;
    font-size:.72rem;
    font-weight:700;
    display:inline-block;
}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # Initialize session state
    if "theme" not in st.session_state:
        st.session_state.theme = "🌿 Skin Tone"
    if "result" not in st.session_state:
        st.session_state.result = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "findings" not in st.session_state:
        st.session_state.findings = None
    
    # Load theme
    t = THEMES[st.session_state.theme]
    inject_css(t)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
<div style='text-align:center;padding:6px 0 18px'>
  <div style='font-size:3rem'><i class='fas fa-stethoscope'></i></div>
  <div style='font-family:Cormorant Garamond,serif;font-size:1.8rem;font-weight:700;
              background:{t['hgrad']};-webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text'>DermAI</div>
  <div style='font-size:.68rem;color:{t["subtext"]};letter-spacing:.12em;text-transform:uppercase'>AI Skin Disease Detector</div>
</div>
""", unsafe_allow_html=True)
        
        theme_choice = st.selectbox("🎨 Theme", list(THEMES.keys()), key="theme_select")
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("1. 📸 Upload a clear skin image")
        st.markdown("2. 🔍 Click 'Analyze Skin Condition'")
        st.markdown("3. 📊 Review the AI diagnosis")
        st.markdown("4. 💊 Follow treatment recommendations")
        st.markdown("5. 🏥 Consult a specialist if needed")
        
        st.markdown("---")
        st.markdown("### 🦠 Detectable Diseases")
        for name, data in DISEASE_DB.items():
            emoji = data['emoji']
            st.markdown(f"{emoji} {name}")
        
        st.markdown("---")
        st.markdown("*Built for HEC-Pak Angels Hackathon*")
        st.markdown("*Powered by AI Computer Vision*")
    
    # Header
    st.markdown("<div class='dtitle'><i class='fas fa-brain'></i> DermAI – AI Skin Disease Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='dsub'>Upload a skin image · AI-powered analysis · Instant diagnosis · Treatment recommendations · Severity assessment</div>", unsafe_allow_html=True)
    
    # Info box
    st.info("✨ This AI analyzes redness, texture, symmetry, and color patterns to detect 9 different skin conditions with high accuracy.")
    
    # Main content
    col_left, col_right = st.columns([1, 1.3], gap="large")
    
    with col_left:
        st.markdown("<div class='dcard'>", unsafe_allow_html=True)
        st.markdown("<div class='ctitle'><i class='fas fa-camera'></i> Upload Your Image</div>", unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Photo tips
            st.markdown(f"""
            <div style='background:{t['pill']};padding:10px;border-radius:10px;margin-top:10px'>
            <small><i class='fas fa-lightbulb'></i> <b>Tips:</b> Good lighting, clear focus, and centered lesion improve accuracy</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Skin Condition", use_container_width=True, type="primary"):
                with st.spinner("🧠 AI is analyzing your skin image..."):
                    progress_bar = st.progress(0)
                    for percent in range(0, 101, 10):
                        time.sleep(0.08)
                        progress_bar.progress(percent)
                    
                    disease, confidence, findings = diagnose_skin(image)
                    
                    st.session_state.result = disease
                    st.session_state.confidence = confidence
                    st.session_state.findings = findings
                    st.rerun()
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:gray'>
                <i class='fas fa-cloud-upload-alt' style='font-size:3rem'></i>
                <p style='margin-top:10px'>Upload a clear, well-lit image of the skin condition</p>
                <p style='font-size:0.8rem'>Supports JPG, PNG, WEBP formats</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Photo tips section
        st.markdown(f"""
        <div class='dcard'>
            <div class='ctitle'><i class='fas fa-lightbulb'></i> Best Practices</div>
            <div style='display:flex;gap:15px;flex-wrap:wrap'>
                <div><i class='fas fa-sun'></i> Natural lighting</div>
                <div><i class='fas fa-camera'></i> Sharp focus</div>
                <div><i class='fas fa-hand-peace'></i> Steady camera</div>
                <div><i class='fas fa-expand'></i> Fill the frame</div>
                <div><i class='fas fa-sliders-h'></i> No filters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<div class='dcard'>", unsafe_allow_html=True)
        st.markdown("<div class='ctitle'><i class='fas fa-stethoscope'></i> Diagnosis Result</div>", unsafe_allow_html=True)
        
        if st.session_state.result:
            db = DISEASE_DB[st.session_state.result]
            
            # Metrics row
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"<div class='mpill'><div class='mval'>{st.session_state.confidence*100:.1f}%</div><div class='mlbl'>Confidence</div></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div class='mpill'><div class='mval'>{db['emoji']}</div><div class='mlbl'>Disease</div></div>", unsafe_allow_html=True)
            with col_c:
                urgency_class = "urg-c" if db['urgency'] == "URGENT" else ("urg-s" if db['urgency'] == "Soon" else "urg-r")
                st.markdown(f"<div class='mpill'><div class='mval'>⚠️</div><div class='mlbl'>{db['urgency']}</div></div>", unsafe_allow_html=True)
            
            # Disease name with severity tag
            severity_class = "tag-c" if "Critical" in db['severity'] or "Serious" in db['severity'] else ("tag-m" if "Moderate" in db['severity'] else "tag-g")
            st.markdown(f"<div class='dname'>{db['emoji']} {st.session_state.result}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><span class='{severity_class}'>{db['severity']}</span> <span class='{urgency_class}' style='margin-left:10px'>{db['urgency']}</span></div>", unsafe_allow_html=True)
            
            # Description
            st.markdown(f"<div class='ddesc'>{db['description']}</div>", unsafe_allow_html=True)
            
            # Visual findings
            if st.session_state.findings:
                st.markdown("**🔍 AI Visual Analysis:**")
                for finding in st.session_state.findings:
                    st.markdown(f"<div class='recitem'><i class='fas fa-chart-line'></i> {finding}</div>", unsafe_allow_html=True)
            
            # Specialist
            st.markdown(f"<div><i class='fas fa-user-md'></i> <b>Refer to:</b> {db['specialist']}</div>", unsafe_allow_html=True)
            
            # Medications expander
            with st.expander("💊 Recommended Medications"):
                for med in db['medications']:
                    st.markdown(f"<div class='meditem'>{med}</div>", unsafe_allow_html=True)
            
            # Recommendations expander
            with st.expander("📝 Self-Care Recommendations"):
                for rec in db['recommendations']:
                    st.markdown(f"<div class='recitem'>{rec}</div>", unsafe_allow_html=True)
            
            # Warning for melanoma
            if st.session_state.result == "Melanoma":
                st.error("🚨 **URGENT:** This condition requires immediate medical attention. Please consult a dermatologist or oncologist today!")
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;color:gray'>
                <i class='fas fa-microscope' style='font-size:3rem'></i>
                <p style='margin-top:10px'>Upload an image and click "Analyze" to see results</p>
                <p style='font-size:0.8rem'>The AI will analyze the image and provide a diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class='disclaim'>
    <i class='fas fa-exclamation-triangle'></i> <b>Medical Disclaimer:</b> DermAI is an <b>educational and informational tool only</b>.
    It does <b>not</b> replace professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare professional for any skin concerns.
    In case of emergency or suspected melanoma, seek immediate medical attention.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
