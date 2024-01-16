def results_to_json(results, model):
    """Converts yolo model output to json (list of dicts)
    This code is from: https://github.com/WelkinU/yolov5-fastapi-demo/blob/main/server.py
    """
    detections = [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxy
    ]

    # Menambahkan informasi jumlah benda yang terdeteksi
    num_objects = sum(len(result) for result in results.xyxy)

    return {
        "detections": detections,
        "num_objects": num_objects,
    }
