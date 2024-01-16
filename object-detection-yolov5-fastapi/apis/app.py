import os
import logging
from io import BytesIO
from typing import List
from warnings import filterwarnings, simplefilter
import ssl
import torch
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from object_detection_yolov5.core.base_detector import ObjectDetector

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

filterwarnings("ignore")
simplefilter(action="ignore", category=FutureWarning)

if not os.path.exists("../logs"):
    os.mkdir("../logs")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler()
file_handler = logging.FileHandler("../logs/api.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app: FastAPI = FastAPI()

model = torch.hub.load("ultralytics/yolov5", "yolov5s")


@app.post("/object_detect")
async def image_detect(request: Request, input_file: UploadFile = File(...)):

    if request.method == "POST":
        try:
            # Membaca gambar dari file yang diunggah
            image: Image = Image.open(BytesIO(await input_file.read()))

            # Membuat objek detector
            ob: ObjectDetector = ObjectDetector(image, model)

            # Melakukan deteksi objek
            json_results = ob.object_detect()

            # Cetak isi dari json_results
            print("json_results:", json_results)

            # Menghitung jumlah barang berdasarkan klasifikasi
            count_by_class = {"CROISSANT": 0, "GOLDENFIL": 0, "MILK": 0}  # Sesuaikan dengan nama kelas sesuai hasil cetakan json_results
            for detection_list in json_results['detections']:
                for result in detection_list:
                    class_name = result["class_name"]
                    if class_name in count_by_class:
                        count_by_class[class_name] += 1

# ...


            # Menghitung total barang yang terdeteksi
            total_objects = sum(count_by_class.values())

            # Log informasi deteksi
            logger.info(["object detection results", json_results])

            return JSONResponse(
                {
                    "data": json_results,
                    "count_by_class": count_by_class,
                    "total_objects": total_objects,
                    "message": "object detected successfully",
                    "errors": None,
                    "status": 200,
                },
                status_code=200,
            )
        except Exception as error:
            # Tangani kesalahan
            print(f"Error: {error}")
            exception_details = str(error)
            print(f"Exception details: {exception_details}")
            return JSONResponse(
                {
                    "message": "object detection failed",
                    "errors": "error",
                    "status": 400,
                },
                status_code=400,
            )

# ...


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
