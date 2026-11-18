import urllib
import os

def download_model_if_missing(model_path="face_landmarker.task"):
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")