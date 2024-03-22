from roboflow import Roboflow
from djitellopy import Tello
import cv2

# Configuración de Roboflow
rf = Roboflow(api_key="tVYeBTfHjApBrSJzNiQF")
project = rf.workspace().project("tfg-ls1lh")
model = project.version(2).model

# Parámetros para el desplazamiento del dron (ajustar según sea necesario)
fps = 30
target_distance = 100  # Distancia total que recorrerá el dron (ajustar según sea necesario)
latas_detectadas = set()

# Función para mostrar el video del dron Tello en la computadora
def mostrar_video_tello():
    # Conectar con el dron Tello
    tello = Tello()
    tello.connect()
    tello.streamon() #Iniciar transmición de video

    cap = cv2.VideoCapture('udp://@0.0.0.0:11111')

    # Ciclo principal para mostrar el video
    while True:
        # Capturar un frame de la transmisión de video
        ret, frame = cap.read()

        # Mostrar el frame en una ventana
        cv2.imshow('Tello Video', frame)

        # Esperar y verificar si se presiona la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Detener la transmisión de video y cerrar la ventana
    tello.streamoff()
    cap.release()
    cv2.destroyAllWindows()


# Función para el desplazamiento y detección de latas con el dron Tello
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
    while True:
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

        # Obtener la distancia al objeto frente al dron (TOF - Time of Flight)
        distance_tof = tello.get_distance_tof()

        # Esperar un breve momento y verificar si se presiona la tecla 'q' para salir
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        # Verificar si ha alcanzado la distancia objetivo y aterrizar
        if distance_tof is not None and distance_tof < target_distance:
            print("Objetivo alcanzado. Aterrizando...")
            tello.land()
            break

    # Cerrar la conexión con el dron
    tello.end()

    # Mostrar la cantidad total de latas detectadas
    print(f"Total de latas de Coca-Cola detectadas: {len(latas_detectadas)}")

# Función principal para seleccionar la cámara
def main():
    choice = input("Selecciona la cámara (tello/windows): ").lower()

    if choice == "tello":
        mostrar_video_tello()  # Mostrar video del dron Tello


# Llamar a la función main al final del script
if __name__ == "__main__":
    main()