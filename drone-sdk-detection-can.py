import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Inicializar el cliente de inferencia
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="tVYeBTfHjApBrSJzNiQF"
)

def process_detection(result, frame):
    # Obtener las dimensiones del fotograma
    height, width, _ = frame.shape

    # Procesar las predicciones
    for prediction in result['predictions']:
        x = int(prediction["x"])
        y = int(prediction["y"])
        w = int(prediction["width"])
        h = int(prediction["height"])
        confidence = prediction["confidence"]
        class_name = prediction["class"]

        # Dibujar el cuadro delimitador y mostrar la etiqueta de clase
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el fotograma con las predicciones
    cv2.imshow('Object Detection', frame)
    cv2.waitKey(1)

# Bucle principal
while True:
    # Capturar un fotograma de la cámara Tello (código no proporcionado)
    frame = capture_frame_from_tello()

    # Realizar la inferencia en el fotograma
    result = CLIENT.infer(frame, model_id="tfg-ls1lh/3")

    # Procesar la detección de objetos y mostrar el resultado en la pantalla
    process_detection(result, frame)
