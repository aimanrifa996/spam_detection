from pathlib import Path
import html
import re

import joblib
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_detection_model.pkl"


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str, modified_time: float):
    """Load the existing serialized sklearn pipeline."""
    return joblib.load(model_path)


def get_model():
    if not MODEL_PATH.exists():
        return None, (
            "The trained model file was not found. Place "
            "`spam_detection_model.pkl` in the same folder as `app.py`."
        )

    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            return None, "The saved model must support predict() and predict_proba()."
        if not hasattr(model, "classes_"):
            return None, "The saved model does not expose its trained class labels."
        return model, None
    except Exception as exc:
        return None, f"The model could not be loaded: {exc}"


# ---------------------------------------------------------
# SESSION STATE AND CALLBACKS
# ---------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "message_input" not in st.session_state:
    st.session_state.message_input = ""


def clear_previous_report():
    # Any edit means the previous report no longer describes the current text.
    st.session_state.analysis = None


def clear_all():
    st.session_state.message_input = ""
    st.session_state.analysis = None


# ---------------------------------------------------------
# CUSTOM UI
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #14213D;
        --slate: #263B65;
        --sky: #38BDF8;
        --background: #F1F5F9;
        --white: #FFFFFF;
        --green: #16A085;
        --red: #E85D75;
        --muted: #64748B;
        --border: #E2E8F0;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--navy);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    .block-container {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    hr { display: none !important; }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #14213D 0%, #263B65 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 23px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 14px 34px rgba(20,33,61,.13);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -65px;
        top: -105px;
        border: 1px solid rgba(56,189,248,.22);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(56,189,248,.035),
                    0 0 0 48px rgba(56,189,248,.025);
        pointer-events: none;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 17px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: rgba(56,189,248,.13);
        border: 1px solid rgba(56,189,248,.28);
        font-size: 25px;
    }

    .brand-name {
        color: #FFFFFF;
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -.5px;
    }

    .machine-tag {
        display: inline-block;
        color: #BAE6FD;
        background: rgba(56,189,248,.12);
        border: 1px solid rgba(56,189,248,.22);
        border-radius: 30px;
        padding: 6px 11px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.1px;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-size: clamp(25px, 4vw, 35px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.9px;
        margin: 0 0 10px 0;
    }

    .hero-copy {
        color: #D8E4F2 !important;
        font-size: 14px;
        line-height: 1.7;
        max-width: 690px;
        margin: 0;
    }

    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 25px 27px;
        box-shadow: 0 5px 18px rgba(20,33,61,.045);
        margin-bottom: 19px;
    }

    .panel-heading {
        color: var(--navy);
        font-size: 20px;
        line-height: 1.3;
        font-weight: 800;
        margin: 0 0 5px 0;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin: 0 0 17px 0;
    }

    .stTextArea textarea {
        background: #F8FAFC !important;
        color: #14213D !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
        min-height: 165px !important;
    }

    .stTextArea textarea:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,.13) !important;
    }

    .stTextArea label { display: none; }

    .counter-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: #64748B;
        font-size: 12px;
        margin: 2px 2px 14px;
    }

    .stButton button {
        min-height: 46px;
        border-radius: 11px !important;
        font-weight: 700 !important;
        border: 1px solid transparent !important;
        transition: all .18s ease;
    }

    .stButton button[kind="primary"] {
        background: #263B65 !important;
        color: #FFFFFF !important;
    }

    .stButton button[kind="primary"]:hover {
        background: #38BDF8 !important;
        color: #14213D !important;
        border-color: #38BDF8 !important;
    }

    .stButton button[kind="secondary"] {
        background: #EDF2F7 !important;
        color: #263B65 !important;
        border-color: #DCE5EF !important;
    }

    .stButton button[kind="secondary"]:hover {
        background: #E2E8F0 !important;
        border-color: #CBD5E1 !important;
    }

    .result {
        border-radius: 16px;
        padding: 22px 24px;
        margin: 4px 0 17px;
    }

    .result.spam {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-left: 5px solid #E85D75;
    }

    .result.ham {
        background: #EAF8F3;
        border: 1px solid #B8E8D8;
        border-left: 5px solid #16A085;
    }

    .result-title {
        font-size: 23px;
        line-height: 1.3;
        font-weight: 800;
        margin: 0 0 6px;
    }

    .result.spam .result-title, .result.spam .result-copy { color: #9F1239; }
    .result.ham .result-title, .result.ham .result-copy { color: #126653; }

    .result-copy { font-size: 13px; line-height: 1.6; margin: 0; }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: 19px 20px;
        min-height: 104px;
        box-shadow: 0 4px 14px rgba(20,33,61,.035);
    }

    .metric-label {
        color: #64748B;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .9px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #14213D;
        font-size: 24px;
        font-weight: 800;
        margin-top: 9px;
        overflow-wrap: anywhere;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: #263B65;
        font-size: 13px;
        font-weight: 700;
        margin: 17px 0 8px;
    }

    .prob-track {
        height: 11px;
        overflow: hidden;
        background: #E8EEF5;
        border-radius: 20px;
    }

    .prob-fill {
        height: 100%;
        border-radius: 20px;
    }

    .prob-fill.spam { background: #E85D75; }
    .prob-fill.ham { background: #16A085; }

    .message-preview {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        color: #263B65;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .notice {
        background: #F0F7FC;
        border: 1px solid #D5EAF7;
        border-radius: 12px;
        padding: 13px 15px;
        color: #38536F;
        font-size: 12px;
        line-height: 1.65;
        margin-top: 17px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 11px;
        line-height: 1.8;
        padding: 16px 0 3px;
    }

    @media (max-width: 640px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .hero { padding: 25px 21px; border-radius: 17px; }
        .panel { padding: 20px 17px; }
        .metric-card { padding: 15px; }
        .metric-value { font-size: 20px; }
        .counter-row { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <section class="hero">
      <div class="brand-row">
        <div class="brand-icon">🛡️</div>
        <div class="brand-name">SpamGuard AI</div>
      </div>
      <div class="machine-tag">MACHINE LEARNING POWERED</div>
      <h1 class="hero-title">Intelligent Message Detection System</h1>
      <p class="hero-copy">
        Identify suspicious messages and protect yourself from unwanted spam
        using Artificial Intelligence.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL STATUS
# ---------------------------------------------------------
model, model_error = get_model()
if model_error:
    st.error(model_error)
    st.info("If the model has not been trained yet, run `python spam_detection.py` once.")
    st.stop()


# ---------------------------------------------------------
# INPUT CARD
# ---------------------------------------------------------
st.markdown(
    """
    <section class="panel">
      <div class="panel-heading">✉️ Message Analyzer</div>
      <p class="panel-subtitle">
        Paste the SMS or email you want to check. Your message will be
        evaluated by the saved machine learning model.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

message = st.text_area(
    "Message",
    placeholder="Paste or type your SMS or email message here...",
    height=165,
    key="message_input",
    on_change=clear_previous_report,
)

word_count = len(message.split())
character_count = len(message)

st.markdown(
    f"""
    <div class="counter-row">
      <span>📝 {word_count} words</span>
      <span>🔤 {character_count} characters</span>
    </div>
    """,
    unsafe_allow_html=True,
)

analyze_col, clear_col = st.columns([3, 1], gap="small")

with analyze_col:
    analyze_clicked = st.button(
        "🔍  Analyze Message",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    st.button(
        "✕  Clear",
        type="secondary",
        use_container_width=True,
        on_click=clear_all,
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if analyze_clicked:
    current_message = st.session_state.message_input

    if not current_message or not current_message.strip():
        st.session_state.analysis = None
        st.warning("Please enter a message before analyzing.")
    else:
        try:
            with st.spinner("Analyzing your message with the trained model..."):
                prediction = model.predict([current_message])[0]
                probabilities = model.predict_proba([current_message])[0]
                classes = list(model.classes_)

                class_lookup = {
                    str(label).strip().lower(): index
                    for index, label in enumerate(classes)
                }

                if "spam" not in class_lookup or "ham" not in class_lookup:
                    raise ValueError(
                        "The model must contain both 'spam' and 'ham' classes. "
                        f"Found: {classes}"
                    )

                spam_probability = float(probabilities[class_lookup["spam"]]) * 100
                ham_probability = float(probabilities[class_lookup["ham"]]) * 100
                predicted_label = str(prediction).strip().lower()

                if predicted_label not in ("spam", "ham"):
                    raise ValueError(f"Unexpected prediction label: {prediction}")

                st.session_state.analysis = {
                    "prediction": predicted_label,
                    "spam_probability": spam_probability,
                    "ham_probability": ham_probability,
                    "confidence": max(spam_probability, ham_probability),
                    "message": current_message,
                }
        except Exception as exc:
            st.session_state.analysis = None
            st.error(f"Unable to analyze this message: {exc}")


# ---------------------------------------------------------
# ANALYSIS REPORT
# ---------------------------------------------------------
report = st.session_state.analysis

if report:
    st.markdown(
        """
        <div style="margin: 27px 0 13px;">
          <div class="panel-heading">📊 Analysis Report</div>
          <p class="panel-subtitle" style="margin-bottom:0;">
            Results generated from your submitted message and the trained model.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_spam = report["prediction"] == "spam"
    if is_spam:
        st.markdown(
            """
            <div class="result spam">
              <div class="result-title">⚠️ SPAM DETECTED</div>
              <p class="result-copy">
                The trained model classified this message as potentially
                suspicious or unwanted. Review it carefully before taking action.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prediction_text = "SPAM"
    else:
        st.markdown(
            """
            <div class="result ham">
              <div class="result-title">✓ NOT SPAM</div>
              <p class="result-copy">
                The trained model classified this message as not spam.
                This prediction does not guarantee that the message is safe.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prediction_text = "NOT SPAM"

    metric1, metric2, metric3 = st.columns(3, gap="medium")

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Predicted Classification</div>
              <div class="metric-value">{prediction_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Model Confidence</div>
              <div class="metric-value">{report['confidence']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        report_words = len(report["message"].split())
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Message Words</div>
              <div class="metric-value">{report_words}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel" style="margin-top:19px;">
          <div class="panel-heading">📈 Probability Breakdown</div>
          <p class="panel-subtitle" style="margin-bottom:5px;">
            Class probabilities returned by the model's predict_proba() method.
          </p>
        """,
        unsafe_allow_html=True,
    )

    spam_pct = max(0.0, min(100.0, report["spam_probability"]))
    ham_pct = max(0.0, min(100.0, report["ham_probability"]))

    st.markdown(
        f"""
          <div class="prob-label">
            <span>🔴 Spam Probability</span>
            <span>{spam_pct:.2f}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill spam" style="width:{spam_pct:.6f}%;"></div>
          </div>
          <div class="prob-label">
            <span>🟢 Not Spam Probability</span>
            <span>{ham_pct:.2f}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ham" style="width:{ham_pct:.6f}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
          <div class="panel-heading">📄 Analyzed Message</div>
          <p class="panel-subtitle">
            Exact text submitted for this classification.
          </p>
        """,
        unsafe_allow_html=True,
    )

    safe_message = html.escape(report["message"])
    st.markdown(
        f'<div class="message-preview">{safe_message}</div></div>',
        unsafe_allow_html=True,
    )

    analyzed_message = report["message"]
    total_characters = len(analyzed_message)
    total_words = len(analyzed_message.split())
    sentence_count = len(re.findall(r"[.!?]+", analyzed_message))

    st.markdown(
        """
        <div class="panel">
          <div class="panel-heading">🔎 Message Statistics</div>
          <p class="panel-subtitle" style="margin-bottom:15px;">
            Basic text statistics for the analyzed message.
          </p>
        """,
        unsafe_allow_html=True,
    )

    stat1, stat2, stat3 = st.columns(3, gap="medium")
    stat1.metric("Total Characters", total_characters)
    stat2.metric("Total Words", total_words)
    stat3.metric("Sentence Count", sentence_count)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notice">
          <strong>Safety notice:</strong> This model's prediction is not a guarantee
          that a message is safe or harmful. Be cautious with unfamiliar links,
          unexpected prizes, payment requests, and messages asking for passwords,
          OTPs, or personal information.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="panel" style="margin-top:24px;">
          <div class="panel-heading">How to use SpamGuard AI</div>
          <p class="panel-subtitle" style="margin-bottom:0;">
            <strong>01 — Enter your message</strong><br>
            Type or paste the SMS or email you want to check.<br><br>
            <strong>02 — Analyze</strong><br>
            Select Analyze Message to run the trained classifier.<br><br>
            <strong>03 — Review your report</strong><br>
            View the prediction, model confidence, class probabilities,
            submitted message, and text statistics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
      <strong>🛡️ SpamGuard AI</strong><br>
      Python · Streamlit · Scikit-learn · TF-IDF · Multinomial Naive Bayes<br>
      Artificial Intelligence &amp; Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
