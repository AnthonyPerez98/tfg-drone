from roboflow import Roboflow
from djitellopy import Tello
from io import BytesIO
import cv2
import os

# Configuración de Roboflow
rf = Roboflow(api_key="tVYeBTfHjApBrSJzNiQF")
project = rf.workspace().project("tfg-ls1lh")
model = project.version(2).model

# Parámetros para el desplazamiento de la cámara (ajustar según sea necesario)
fps = 30
target_distance = 100  # Distancia total que recorrerá la cámara (ajustar según sea necesario)
latas_detectadas = set()

def desplazamiento_camara_dron():
    global latas_detectadas

    # Conectar con el dron Tello
    tello = Tello()
    tello.connect()
    tello.takeoff()

    # Iniciar el recorrido en forma de "L"
    tello.move_forward(50)
    tello.move_right(50)

    # Capturar la cámara del dron y realizar el desplazamiento
    while tello.get_distance_x() < target_distance:
        # Capturar el frame de la cámara del dron
        frame = tello.get_frame_read().frame

        # Realizar inferencia con el modelo de Roboflow
        response = model.predict(frame, confidence=40, overlap=30).json()

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

    # Aterrizar el dron al final del recorrido
    tello.land()

    # Cerrar la conexión con el dron
    tello.end()

# Función para la cámara de Windows
def desplazamiento_camara_windows():
    global latas_detectadas

    # Inicializar la captura de la cámara de Windows
    cap = cv2.VideoCapture(0)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    # Capturar la cámara y realizar el desplazamiento
    while True:
        # Capturar el frame de la cámara
        ret, frame = cap.read()

        # Realizar inferencia con el modelo de Roboflow
        response = model.predict(frame, confidence=40, overlap=30).json()

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

        # Esperar un breve momento y verificar si se presiona la tecla 'q' para salir
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Liberar la captura y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

# Función principal para seleccionar la cámara
def main():
    choice = input("Selecciona la cámara (windows/tello): ").lower()

    if choice == "windows":
        desplazamiento_camara_windows()
    elif choice == "tello":
        desplazamiento_camara_dron()
    else:
        print("Opción no válida. Selecciona 'windows' o 'tello'.")

# Ejecutar la función principal
main()

# Mostrar la cantidad total de latas detectadas
print(f"Total de latas de Coca-Cola detectadas: {len(latas_detectadas)}")
