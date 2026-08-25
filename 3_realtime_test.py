import os
import cv2
import pickle
import time
import threading

import numpy as np
import pyttsx3

from skimage.feature import hog


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640

CAMERA_HEIGHT = 480

ROI_SIZE = 300

IMAGE_SIZE = 128

STABLE_FRAMES = 7

MIN_STABLE_COUNT = 5

VOICE_COOLDOWN = 2.0


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {

    "0": "Hello",

    "1": "Thank You",

    "2": "Help",

    "3": "No Sign"
}


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
# MODEL PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(

    os.path.abspath(
        __file__
    )
)


MODEL_PATH = os.path.join(

    SCRIPT_DIR,

    "model.p"
)


# ============================================================
# HEADER
# ============================================================

print()
print("======================================================")
print("             ISL REAL-TIME TRANSLATOR")
print("======================================================")
print()


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    print(
        "ERROR: model.p not found!"
    )

    print(
        MODEL_PATH
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


with open(

    MODEL_PATH,

    "rb"

) as file:

    package = pickle.load(
        file
    )


model = package["model"]


IMAGE_SIZE = package.get(
    "image_size",
    128
)


print(
    "MODEL LOADED SUCCESSFULLY"
)

print(
    "Feature type:",
    package.get(
        "feature_type",
        "HOG"
    )
)


# ============================================================
# HOG FUNCTION
# ============================================================

def extract_hog(image):

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    if len(image.shape) == 3:

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY
        )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(

        image,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Contrast normalization
    # --------------------------------------------------------

    image = cv2.equalizeHist(
        image
    )


    # --------------------------------------------------------
    # HOG
    # --------------------------------------------------------

    features = hog(

        image,

        orientations=9,

        pixels_per_cell=(8, 8),

        cells_per_block=(2, 2),

        block_norm="L2-Hys",

        feature_vector=True
    )


    return features.astype(
        np.float32
    )


# ============================================================
# CAMERA
# ============================================================

print()
print(
    "Opening camera..."
)


cap = cv2.VideoCapture(

    CAMERA_INDEX,

    cv2.CAP_DSHOW
)


if not cap.isOpened():

    cap.release()

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


if not cap.isOpened():

    print()
    print(
        "ERROR: Camera could not be opened!"
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    CAMERA_WIDTH
)

cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    CAMERA_HEIGHT
)


print(
    "Camera opened successfully."
)


# ============================================================
# TTS
# ============================================================

print()
print(
    "Initializing TTS..."
)


tts_available = False


try:

    test_engine = pyttsx3.init()

    test_engine.setProperty(
        "rate",
        150
    )

    test_engine.setProperty(
        "volume",
        1.0
    )

    voices = test_engine.getProperty(
        "voices"
    )

    print(
        "Available voices:",
        len(voices)
    )

    test_engine.stop()

    del test_engine

    tts_available = True

    print(
        "TTS initialized successfully."
    )

except Exception as e:

    print(
        "TTS ERROR:",
        e
    )


# ============================================================
# SPEECH
# ============================================================

speech_lock = threading.Lock()


def speak(text):

    if not tts_available:

        return


    with speech_lock:

        try:

            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                150
            )

            engine.setProperty(
                "volume",
                1.0
            )

            engine.say(
                text
            )

            engine.runAndWait()

            engine.stop()

            del engine

        except Exception as e:

            print(
                "Speech error:",
                e
            )


def speak_async(text):

    thread = threading.Thread(

        target=speak,

        args=(text,),

        daemon=True
    )

    thread.start()


# ============================================================
# VARIABLES
# ============================================================

prediction_history = []

confirmed_class = "3"

last_spoken_class = None

last_spoken_time = 0

frame_counter = 0

has_seen_no_sign = False


# ============================================================
# CAMERA READY
# ============================================================

print()
print("======================================================")
print("                    CAMERA READY")
print("======================================================")
print()
print("0 -> Hello")
print("1 -> Thank You")
print("2 -> Help")
print("3 -> No Sign")
print()
print("Put your hand INSIDE the GREEN BOX.")
print()
print("Press Q to quit.")
print()
print("======================================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        continue


    # --------------------------------------------------------
    # Mirror
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    # ========================================================
    # ROI
    # ========================================================

    x1 = int(
        (width - ROI_SIZE) / 2
    )

    y1 = int(
        (height - ROI_SIZE) / 2
    )

    x2 = x1 + ROI_SIZE

    y2 = y1 + ROI_SIZE


    roi = frame[
        y1:y2,
        x1:x2
    ]


    # ========================================================
    # HOG
    # ========================================================

    features = extract_hog(
        roi
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction = model.predict(

            features.reshape(
                1,
                -1
            )
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        break


    predicted_class = str(
        prediction[0]
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    confidence = 0.0

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(

                features.reshape(
                    1,
                    -1
                )
            )[0]


            confidence = float(
                np.max(
                    probabilities
                )
            )

        except:

            confidence = 0.0


    # ========================================================
    # HISTORY
    # ========================================================

    prediction_history.append(
        predicted_class
    )


    if len(
        prediction_history
    ) > STABLE_FRAMES:

        prediction_history.pop(
            0
        )


    # ========================================================
    # MAJORITY VOTE
    # ========================================================

    stable_prediction = predicted_class

    stable_count = 1


    if len(
        prediction_history
    ) >= STABLE_FRAMES:

        counts = {}


        for value in prediction_history:

            counts[value] = (
                counts.get(
                    value,
                    0
                )
                + 1
            )


        stable_prediction = max(

            counts,

            key=counts.get
        )


        stable_count = counts[
            stable_prediction
        ]


        if stable_count >= MIN_STABLE_COUNT:

            confirmed_class = (
                stable_prediction
            )


    # ========================================================
    # TRANSLATION
    # ========================================================

    translation = TRANSLATIONS.get(

        confirmed_class,

        TRANSLATIONS["3"]
    )


    english = translation[
        "english"
    ]

    bengali = translation[
        "bengali"
    ]

    hindi = translation[
        "hindi"
    ]


    # ========================================================
    # TERMINAL DEBUG
    # ========================================================

    frame_counter += 1


    if frame_counter % 15 == 0:

        print(

            f"Raw={predicted_class} "
            f"({CLASS_NAMES[predicted_class]}) "
            f"| Confidence={confidence * 100:.1f}% "
            f"| Stable={confirmed_class} "
            f"| StableCount={stable_count}"
        )


    # ========================================================
    # VOICE LOGIC
    # ========================================================

    current_time = time.time()


    # --------------------------------------------------------
    # NO SIGN
    # --------------------------------------------------------

    if confirmed_class == "3":

        has_seen_no_sign = True

        last_spoken_class = None


    # --------------------------------------------------------
    # SIGN
    # --------------------------------------------------------

    elif confirmed_class in [
        "0",
        "1",
        "2"
    ]:

        # ----------------------------------------------------
        # Important:
        # Must first see No Sign before speaking.
        #
        # This prevents:
        #
        # Camera starts
        # -> model says Hello
        # -> immediately speaks Hello
        # ----------------------------------------------------

        if has_seen_no_sign:

            # ------------------------------------------------
            # New sign
            # ------------------------------------------------

            if (

                confirmed_class
                !=
                last_spoken_class

            ) and (

                current_time
                -
                last_spoken_time
                >=
                VOICE_COOLDOWN
            ):

                print()
                print("--------------------------------------")

                print(
                    "SIGN DETECTED:",
                    english
                )

                print(
                    "Bengali:",
                    bengali
                )

                print(
                    "Hindi:",
                    hindi
                )

                print(
                    "Confidence:",
                    f"{confidence * 100:.1f}%"
                )

                print("--------------------------------------")


                speak_async(
                    english
                )


                last_spoken_class = (
                    confirmed_class
                )

                last_spoken_time = (
                    current_time
                )


    # ========================================================
    # GREEN ROI
    # ========================================================

    cv2.rectangle(

        frame,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        3
    )


    # ========================================================
    # INFORMATION BOX
    # ========================================================

    cv2.rectangle(

        frame,

        (10, 10),

        (445, 180),

        (0, 0, 0),

        -1
    )


    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    cv2.putText(

        frame,

        "Raw: " + CLASS_NAMES[predicted_class],

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

        (255, 255, 255),

        2
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Confidence: {confidence * 100:.1f}%",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (255, 255, 0),

        2
    )


    # --------------------------------------------------------
    # Stable
    # --------------------------------------------------------

    cv2.putText(

        frame,

        "Detected: " + english,

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

        (0, 255, 0),

        2
    )


    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Stable: {stable_count}/{STABLE_FRAMES}",

        (20, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # --------------------------------------------------------
    # Voice status
    # --------------------------------------------------------

    voice_status = (
        "VOICE READY"
        if has_seen_no_sign
        else "SHOW NO SIGN FIRST"
    )


    cv2.putText(

        frame,

        voice_status,

        (20, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        (0, 255, 255),

        2
    )


    # ========================================================
    # MODEL INPUT PREVIEW
    # ========================================================

    gray_preview = cv2.cvtColor(

        roi,

        cv2.COLOR_BGR2GRAY
    )


    gray_preview = cv2.resize(

        gray_preview,

        (160, 160),

        interpolation=cv2.INTER_AREA
    )


    gray_preview = cv2.equalizeHist(
        gray_preview
    )


    gray_preview = cv2.cvtColor(

        gray_preview,

        cv2.COLOR_GRAY2BGR
    )


    preview_x = width - 175

    preview_y = 10


    frame[
        preview_y:preview_y + 160,
        preview_x:preview_x + 160
    ] = gray_preview


    cv2.rectangle(

        frame,

        (
            preview_x,
            preview_y
        ),

        (
            preview_x + 160,
            preview_y + 160
        ),

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        "MODEL INPUT",

        (
            preview_x,
            preview_y + 180
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # ========================================================
    # QUIT
    # ========================================================

    cv2.putText(

        frame,

        "Q = Quit",

        (
            20,
            height - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "ISL Real-Time Translator",

        frame
    )


    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

cv2.waitKey(1)


print()
print("======================================================")
print("             ISL TRANSLATOR STOPPED")
print("======================================================")