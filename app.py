from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
import os
import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama
import urllib.request

app = Flask(__name__)

# Configuration for file uploads
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
configure_uploads(app, photos)

# Path for saving results
app.config["RESULTS_FOLDER"] = "results"

# Initialize SimpleLama instance
simple_lama = SimpleLama()


def download_yolov4_weights():
    if not os.path.exists("yolov4.weights"):
        print("Downloading YOLOv4 weights...")
        urllib.request.urlretrieve(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            "yolov4.weights")
        print("Download complete.")


def detect_objects_yolov4(image_array):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    classes = []

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    height, width, _ = image_array.shape
    blob = cv2.dnn.blobFromImage(image_array, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x, y, w, h = box
        mask[y:y + h, x:x + w] = 255

    return mask


def inpaint_image(image_array, mask):
    # Convert NumPy arrays to PIL Images
    image_pil = Image.fromarray(image_array)
    mask_pil = Image.fromarray(mask)

    # Perform inpainting using simple-lama-inpainting
    result_pil = simple_lama(image_pil, mask_pil)

    # Convert the result back to NumPy array
    result_array = np.array(result_pil)
    return result_array


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo.filename != "":
            photo_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], photo.filename)
            photo.save(photo_path)

            image_array = cv2.imread(photo_path)
            mask = detect_objects_yolov4(image_array)
            inpainted_image = inpaint_image(image_array, mask)

            result_filename = f"result_{photo.filename}"
            result_path = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
            cv2.imwrite(result_path, inpainted_image)

            return render_template("result.html", result_filename=result_filename)

    return render_template("index.html")


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    download_yolov4_weights()
    app.run(debug=False)
