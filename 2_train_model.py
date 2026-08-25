import os
import cv2
import pickle
import numpy as np

from skimage.feature import hog

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "./data"

MODEL_FILE = "model.p"

IMAGE_SIZE = 128


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
# HEADER
# ============================================================

print()
print("======================================================")
print("              ISL HOG + SVM TRAINING")
print("======================================================")
print()


# ============================================================
# HOG FUNCTION
# ============================================================

def extract_hog(image):

    # Make sure image is grayscale

    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


    # Resize

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )


    # Mild contrast normalization

    image = cv2.equalizeHist(
        image
    )


    # HOG

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
# CHECK DATA
# ============================================================

if not os.path.exists(
    DATA_DIR
):

    print(
        "ERROR: data folder not found!"
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

data = []

labels = []


print(
    "Loading images..."
)

print()


for class_name in sorted(

    os.listdir(DATA_DIR),

    key=lambda x: int(x)
    if x.isdigit()
    else x
):

    class_path = os.path.join(
        DATA_DIR,
        class_name
    )


    if not os.path.isdir(
        class_path
    ):
        continue


    if class_name not in CLASS_NAMES:

        continue


    count = 0


    for filename in os.listdir(
        class_path
    ):

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):

            continue


        image_path = os.path.join(
            class_path,
            filename
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            continue


        features = extract_hog(
            image
        )


        data.append(
            features
        )

        labels.append(
            class_name
        )

        count += 1


    print(
        f"Class {class_name} "
        f"({CLASS_NAMES[class_name]}): "
        f"{count} images"
    )


# ============================================================
# NUMPY
# ============================================================

data = np.asarray(
    data,
    dtype=np.float32
)

labels = np.asarray(
    labels
)


# ============================================================
# DATASET INFO
# ============================================================

print()
print("======================================================")
print("DATASET INFORMATION")
print("======================================================")

print(
    "Total images:",
    len(data)
)

print(
    "HOG features:",
    data.shape[1]
)

print()


for class_id in [
    "0",
    "1",
    "2",
    "3"
]:

    print(
        f"{class_id} -> "
        f"{CLASS_NAMES[class_id]}: "
        f"{np.sum(labels == class_id)}"
    )


# ============================================================
# CHECK
# ============================================================

if len(data) < 100:

    print()
    print(
        "ERROR: Not enough training images."
    )

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


# ============================================================
# SPLIT
# ============================================================

print()
print("======================================================")
print("TRAIN / TEST SPLIT")
print("======================================================")


x_train, x_test, y_train, y_test = train_test_split(

    data,

    labels,

    test_size=0.20,

    random_state=42,

    shuffle=True,

    stratify=labels
)


print()
print(
    "Training images:",
    len(x_train)
)

print(
    "Testing images:",
    len(x_test)
)


# ============================================================
# SVM
# ============================================================

print()
print("======================================================")
print("TRAINING SVM...")
print("======================================================")
print()


model = SVC(

    kernel="rbf",

    C=10,

    gamma="scale",

    probability=True,

    class_weight=None,

    random_state=42
)


model.fit(

    x_train,

    y_train
)


print()
print(
    "SVM training completed!"
)


# ============================================================
# TEST
# ============================================================

y_pred = model.predict(
    x_test
)


accuracy = accuracy_score(

    y_test,

    y_pred
)


print()
print("======================================================")
print(
    f"MODEL ACCURACY: {accuracy * 100:.2f}%"
)
print("======================================================")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("CLASSIFICATION REPORT")
print("======================================================")


print(

    classification_report(

        y_test,

        y_pred,

        labels=[
            "0",
            "1",
            "2",
            "3"
        ],

        target_names=[
            "Hello",
            "Thank You",
            "Help",
            "No Sign"
        ],

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("======================================================")


matrix = confusion_matrix(

    y_test,

    y_pred,

    labels=[
        "0",
        "1",
        "2",
        "3"
    ]
)


print()

print(
    "             Hello  ThankYou  Help  NoSign"
)

print(
    "Hello     ",
    matrix[0]
)

print(
    "Thank You ",
    matrix[1]
)

print(
    "Help      ",
    matrix[2]
)

print(
    "No Sign   ",
    matrix[3]
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("======================================================")
print("SAVING MODEL...")
print("======================================================")


model_path = os.path.join(

    os.path.dirname(
        os.path.abspath(__file__)
    ),

    MODEL_FILE
)


model_package = {

    "model": model,

    "image_size": IMAGE_SIZE,

    "feature_type": "HOG",

    "hog_settings": {

        "orientations": 9,

        "pixels_per_cell": (8, 8),

        "cells_per_block": (2, 2),

        "block_norm": "L2-Hys"
    },

    "classes": CLASS_NAMES
}


with open(

    model_path,

    "wb"

) as file:

    pickle.dump(

        model_package,

        file
    )


print()
print("======================================================")
print("          MODEL SAVED SUCCESSFULLY")
print("======================================================")

print()
print(
    model_path
)

print()
print("0 -> Hello")
print("1 -> Thank You")
print("2 -> Help")
print("3 -> No Sign")

print()
print("Training finished!")