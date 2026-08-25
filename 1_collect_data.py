import cv2
import os
import time


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "./data"

NUMBER_OF_CLASSES = 4

IMAGES_PER_CLASS = 300

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

ROI_SIZE = 300


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = {
    0: "HELLO",
    1: "THANK YOU",
    2: "HELP",
    3: "NO SIGN"
}


INSTRUCTIONS = {
    0: "Show HELLO sign",
    1: "Show THANK YOU sign",
    2: "Show HELP sign",
    3: "REMOVE YOUR HAND"
}


# ============================================================
# CREATE DATA FOLDER
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# OPEN CAMERA
# ============================================================

print()
print("======================================================")
print("              ISL DATA COLLECTION")
print("======================================================")
print()

print("Opening camera...")


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
    print("ERROR: Camera could not be opened.")

    input(
        "\nPress Enter to exit..."
    )

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


print("Camera opened successfully.")


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_roi_coordinates(frame):

    height, width = frame.shape[:2]

    x1 = int(
        (width - ROI_SIZE) / 2
    )

    y1 = int(
        (height - ROI_SIZE) / 2
    )

    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    return x1, y1, x2, y2


# ============================================================
# COLLECT EACH CLASS
# ============================================================

for class_id in range(
    NUMBER_OF_CLASSES
):

    class_name = CLASS_NAMES[class_id]

    instruction = INSTRUCTIONS[class_id]

    class_dir = os.path.join(
        DATA_DIR,
        str(class_id)
    )

    os.makedirs(
        class_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Delete old images
    # --------------------------------------------------------

    print()

    print(
        "Preparing class:",
        class_id,
        class_name
    )


    for filename in os.listdir(
        class_dir
    ):

        path = os.path.join(
            class_dir,
            filename
        )

        if os.path.isfile(path):

            try:
                os.remove(path)
            except:
                pass


    # --------------------------------------------------------
    # Instruction
    # --------------------------------------------------------

    print()
    print("======================================================")
    print(
        f"CLASS {class_id} -> {class_name}"
    )
    print("======================================================")

    print(
        instruction
    )

    print()
    print(
        'Press S when ready.'
    )

    print(
        'Press Q to quit.'
    )


    # ========================================================
    # WAIT FOR S
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        x1, y1, x2, y2 = get_roi_coordinates(
            frame
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            f"CLASS {class_id}: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            instruction,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Press S to start",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            "ISL Data Collection",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            raise SystemExit


        if key == ord("s"):

            break


    # ========================================================
    # COUNTDOWN
    # ========================================================

    print()
    print("Starting in...")

    for number in [3, 2, 1]:

        start = time.time()

        while time.time() - start < 1:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(
                frame,
                1
            )

            x1, y1, x2, y2 = get_roi_coordinates(
                frame
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                str(number),
                (
                    frame.shape[1] // 2 - 30,
                    frame.shape[0] // 2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 255),
                4
            )

            cv2.imshow(
                "ISL Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()

                raise SystemExit


    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    print()
    print(
        f"Collecting {IMAGES_PER_CLASS} images..."
    )


    counter = 0


    while counter < IMAGES_PER_CLASS:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        x1, y1, x2, y2 = get_roi_coordinates(
            frame
        )


        # ----------------------------------------------------
        # Extract ROI
        # ----------------------------------------------------

        roi = frame[
            y1:y2,
            x1:x2
        ]


        if roi.size == 0:
            continue


        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        gray = cv2.resize(
            gray,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        filename = os.path.join(
            class_dir,
            f"{counter:04d}.png"
        )


        cv2.imwrite(
            filename,
            gray
        )


        counter += 1


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display = frame.copy()


        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        cv2.putText(
            display,
            f"{class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"Images: {counter}/{IMAGES_PER_CLASS}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "Keep hand inside GREEN BOX",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            "ISL Data Collection",
            display
        )


        key = cv2.waitKey(25) & 0xFF


        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()

            raise SystemExit


    print()
    print(
        f"Class {class_id} completed!"
    )


# ============================================================
# CLOSE
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("======================================================")
print("           DATA COLLECTION COMPLETE")
print("======================================================")
print()
print("0 -> Hello")
print("1 -> Thank You")
print("2 -> Help")
print("3 -> No Sign")
print()
print(
    f"{IMAGES_PER_CLASS} images per class"
)
print()