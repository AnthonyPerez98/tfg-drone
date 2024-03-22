from roboflow import Roboflow
import cv2
import time
import numpy as np
import os

# Configuración de Roboflow
rf = Roboflow(api_key="tVYeBTfHjApBrSJzNiQF")
project = rf.workspace().project("tfg-ls1lh")
model = project.version(2).model

# Configuración del dron Tello
cap = cv2.VideoCapture('udp://@0.0.0.0:11111')

# Parámetros para el desplazamiento de la cámara (ajustar según sea necesario)
fps = 30
target_distance = 100  # Distancia total que recorrerá la cámara (ajustar según sea necesario)
latas_detectadas = set()

def desplazamiento_camara_dron():
    global latas_detectadas

    # Esperar un momento para estabilizar la transmisión de video
    time.sleep(2)

    # Iniciar el tiempo de referencia para el bucle
    start_time = time.time()

    while True:
        # Capturar el frame de la cámara del dron
        ret, frame = cap.read()

        # Crear un nombre de archivo temporal único
        temp_file_path = f"temp_frame_{len(latas_detectadas)}.jpg"

        # Guardar el frame en el archivo temporal
        cv2.imwrite(temp_file_path, frame)

        # Realizar inferencia con el modelo de Roboflow
        response = model.predict(temp_file_path, confidence=40, overlap=30).json()

        # Procesar los resultados de la inferencia
        if "predictions" in response:
            for prediction in response["predictions"]:
                x, y, width, height, class_label = (
                    prediction["x"],
                    prediction["y"],
                    prediction["width"],
                    prediction["height"],
                    prediction["class"],
                )

                # Verificar si la lata ya fue contada
                lata_id = f"{x}_{y}_{width}_{height}_{class_label}"
                if lata_id not in latas_detectadas:
                    # Dibujar un rectángulo alrededor del objeto detectado
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Contar la lata de Coca-Cola detectada
                    latas_detectadas.add(lata_id)

        # Mostrar el frame con las detecciones
        cv2.imshow("Coca-Cola Detection", frame)

        # Esperar un breve momento
        cv2.waitKey(int(1000 / fps))

        # Comprobar si ha pasado el tiempo de vuelo máximo
        elapsed_time = time.time() - start_time
        if elapsed_time > 30:  # 30 segundos de vuelo máximo
            break

    # Cerrar la conexión con el dron
    cap.release()
    cv2.destroyAllWindows()

# Función principal
def main():
    desplazamiento_camara_dron()

    # Mostrar la cantidad total de latas detectadas
    print(f"Total de latas de Coca-Cola detectadas: {len(latas_detectadas)}")

# Ejecutar la función principal
main()
