import streamlit as st
import cv2
import numpy as np
import pickle
import os
import html

# ============================================================
# ISL REAL-TIME TRANSLATOR - WEB PROTOTYPE
# ============================================================

st.set_page_config(
    page_title="ISL Real-Time Translator",
    page_icon="🤟",
    layout="wide"
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.p"
)

IMAGE_SIZE = (64, 64)

MIN_CONFIDENCE = 55.0


# ============================================================
# TRANSLATIONS
# ============================================================

TRANSLATIONS = {
    "0": {
        "english": "Hello",
        "bengali": "নমস্কার",
        "hindi": "नमस्ते"
    },

    "1": {
        "english": "Thank You",
        "bengali": "ধন্যবাদ",
        "hindi": "धन्यवाद"
    },

    "2": {
        "english": "Help",
        "bengali": "সাহায্য",
        "hindi": "मदद"
    },

    "3": {
        "english": "No Sign",
        "bengali": "কোনো সংকেত নেই",
        "hindi": "कोई संकेत नहीं"
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        with open(MODEL_PATH, "rb") as file:
            model_data = pickle.load(file)

        if isinstance(model_data, dict):

            if "model" in model_data:
                return model_data["model"]

        return model_data

    except Exception as error:

        st.error(f"Could not load model: {error}")

        return None


model = load_model()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        border: 2px solid #2ecc71;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-size: 34px;
        font-weight: 800;
    }

    .translation {
        font-size: 24px;
        margin-top: 10px;
    }

    .feature-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #555;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤟 Indian Sign Language Translator</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered Computer Vision prototype for translating
    selected Indian Sign Language gestures.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "model.p was not found. "
        "Please keep model.p in the same folder as web_app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤟 Supported Signs")

    st.write("**0 → Hello**")
    st.write("**1 → Thank You**")
    st.write("**2 → Help**")
    st.write("**3 → No Sign**")

    st.divider()

    st.header("📌 How to Use")

    st.write(
        """
        1. Allow camera access.
        2. Place your hand clearly in front of the camera.
        3. Keep good lighting.
        4. Capture the image.
        5. The AI will predict the sign.
        """
    )

    st.divider()

    st.caption(
        "Model: Random Forest Classifier"
    )

    st.caption(
        "Input: 64 × 64 grayscale image"
    )


# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1.2, 1]
)


# ============================================================
# CAMERA
# ============================================================

with left_column:

    st.subheader("📷 Camera")

    st.info(
        "Allow camera permission and place your hand clearly "
        "inside the camera view."
    )

    camera_image = st.camera_input(
        "Take a picture of your sign"
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_bytes):

    image_array = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return None, 0.0

    # --------------------------------------------------------
    # SAME PREPROCESSING AS TRAINING
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        IMAGE_SIZE
    )

    features = resized.flatten().reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        features
    )

    predicted_class = str(
        prediction[0]
    )

    confidence = 0.0

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = float(
            np.max(probabilities) * 100
        )

    return predicted_class, confidence


# ============================================================
# PROCESS IMAGE
# ============================================================

if camera_image is not None:

    predicted_class, confidence = predict_image(
        camera_image.getvalue()
    )

    if predicted_class is None:

        st.error(
            "Could not read the camera image."
        )

    else:

        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        if confidence < MIN_CONFIDENCE:

            predicted_class = "3"

        result = TRANSLATIONS.get(
            predicted_class,
            TRANSLATIONS["3"]
        )

        # ====================================================
        # RESULT
        # ====================================================

        with right_column:

            st.subheader("🤖 AI Result")

            if predicted_class == "3":

                st.markdown(
                    """
                    <div class="result-box">
                    <div class="result-title">
                    No Sign
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "Please show one of the supported signs."
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box">

                    <div class="result-title">
                    {html.escape(result["english"])}
                    </div>

                    <div class="translation">
                    বাংলা: {html.escape(result["bengali"])}
                    </div>

                    <div class="translation">
                    हिंदी: {html.escape(result["hindi"])}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )

                st.write(
                    f"Confidence: **{confidence:.1f}%**"
                )

                # ------------------------------------------------
                # BROWSER VOICE
                # ------------------------------------------------

                spoken_text = result["english"]

                speech_code = f"""
                <script>

                const text = "{spoken_text}";

                function speakText() {{

                    if ("speechSynthesis" in window) {{

                        window.speechSynthesis.cancel();

                        const speech =
                            new SpeechSynthesisUtterance(text);

                        speech.lang = "en-US";
                        speech.rate = 0.9;
                        speech.pitch = 1.0;
                        speech.volume = 1.0;

                        window.speechSynthesis.speak(
                            speech
                        );
                    }}

                }}

                speakText();

                </script>
                """

                st.components.v1.html(
                    speech_code,
                    height=1
                )

                st.success(
                    f"🔊 Voice: {spoken_text}"
                )


# ============================================================
# SUPPORTED SIGNS
# ============================================================

st.divider()

st.subheader("🤟 Supported ISL Signs")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        ### 🖐️
        **Hello**

        Class 0
        """
    )

with col2:

    st.markdown(
        """
        ### 🙏
        **Thank You**

        Class 1
        """
    )

with col3:

    st.markdown(
        """
        ### 🆘
        **Help**

        Class 2
        """
    )

with col4:

    st.markdown(
        """
        ### 🚫
        **No Sign**

        Class 3
        """
    )


# ============================================================
# PROJECT FEATURES
# ============================================================

st.divider()

st.subheader("🚀 Project Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.markdown(
        """
        <div class="feature-box">

        ### 👁️ Computer Vision

        Webcam-based hand gesture
        recognition using OpenCV.

        </div>
        """,
        unsafe_allow_html=True
    )

with feature2:

    st.markdown(
        """
        <div class="feature-box">

        ### 🤖 Machine Learning

        Random Forest classifier trained
        on ISL gesture images.

        </div>
        """,
        unsafe_allow_html=True
    )

with feature3:

    st.markdown(
        """
        <div class="feature-box">

        ### 🔊 Multilingual Output

        English, Bengali and Hindi
        translation with voice feedback.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.subheader("📖 About This Project")

st.write(
    """
    The Indian Sign Language Real-Time Translator is a
    computer vision and machine learning prototype designed
    to help reduce communication barriers.

    The system recognizes selected hand gestures and converts
    them into text in English, Bengali and Hindi, with
    browser-based voice feedback.

    Currently supported gestures:

    • Hello
    • Thank You
    • Help
    • No Sign

    This is a prototype and the current model supports a
    limited number of gesture classes.
    """
)


# ============================================================
# TEAM
# ============================================================

st.divider()

st.markdown(
    """
    ### 👥 TEAM RUDRASTRA

    **Indian Sign Language (ISL) Real-Time Translator**

    Built using Python, OpenCV, NumPy,
    Scikit-learn and Streamlit.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TEAM RUDRASTRA • ISL Real-Time Translator • Prototype"
)