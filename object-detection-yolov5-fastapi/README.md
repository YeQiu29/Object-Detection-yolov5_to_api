[![ObjectDetectionService](https://github.com/olahsymbo/object-detection-yolov5-fastapi/actions/workflows/object-detection-pipeline.yml/badge.svg)](https://github.com/olahsymbo/object-detection-yolov5-fastapi/actions/workflows/object-detection-pipeline.yml)
# Object Detection

A FastAPI object detection application based on Yolov5 model.

## Getting Started

- Clone the project
- `cd` into the codebase
- run `poetry shell` and `poetry install` to set the virtual environment and install the necessary dependencies

## Start the app
Since we used docker, we can start the app by running these commands:

```
sudo docker-compose build
sudo docker-compose up -d
```
The app will be served on port `8000`

## Make request

An example `curl` request:

```
curl --location '127.0.0.1:8000/object_detect' \
--form 'input_file=@"/Users/Downloads/download.jpeg"'
```

Valid response:
```
{
    "data": [
        [
            {
                "class": 0,
                "class_name": "person",
                "bbox": [
                    134,
                    330,
                    187,
                    579
                ],
                "confidence": 0.7723177671432495
            },
            {
                "class": 0,
                "class_name": "person",
                "bbox": [
                    90,
                    373,
                    140,
                    587
                ],
                "confidence": 0.7619432210922241
            },
            {
                "class": 2,
                "class_name": "car",
                "bbox": [
                    492,
                    314,
                    566,
                    430
                ],
                "confidence": 0.7516090273857117
            },
            {
                "class": 2,
                "class_name": "car",
                "bbox": [
                    379,
                    321,
                    455,
                    460
                ],
                "confidence": 0.7398971915245056
            },
            {
                "class": 0,
                "class_name": "person",
                "bbox": [
                    267,
                    332,
                    315,
                    541
                ],
                "confidence": 0.4890599846839905
            },
            {
                "class": 0,
                "class_name": "person",
                "bbox": [
                    592,
                    322,
                    611,
                    389
                ],
                "confidence": 0.45573562383651733
            },
            {
                "class": 2,
                "class_name": "car",
                "bbox": [
                    444,
                    313,
                    477,
                    400
                ],
                "confidence": 0.38912448287010193
            },
            {
                "class": 0,
                "class_name": "person",
                "bbox": [
                    209,
                    324,
                    260,
                    553
                ],
                "confidence": 0.30747783184051514
            }
        ]
    ],
    "message": "object detected successfully",
    "errors": null,
    "status": 200
}
```

Error response:

```
{
    "message": "object detection failed",
    "errors": "error",
    "status": 400
}
```
