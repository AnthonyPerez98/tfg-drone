
from roboflow import Roboflow
from djitellopy import Tello
import cv2

# Configuración de Roboflow
rf = Roboflow(api_key="tVYeBTfHjApBrSJzNiQF")
project = rf.workspace().project("tfg-ls1lh")
model = project.version(2).model

# Parámetros para la detección y el desplazamiento del dron (ajustar según sea necesario)
fps = 30
latas_detectadas = set()

# Función para procesar la detección y contar las latas
def procesar_deteccion(resultado):
    global latas_detectadas

    if "predictions" in resultado:
        for prediction in resultado["predictions"]:
            x, y, width, height, class_label = (
                int(prediction["x"]),
                int(prediction["y"]),
                int(prediction["width"]),
                int(prediction["height"]),
                prediction["class"],
            )

            # Calcular el área de la detección
            area = width * height

            # Verificar si la lata ya fue contada
            lata_id = f"{x}_{y}_{width}_{height}_{class_label}"
            if lata_id not in latas_detectadas and area > 100:  # Ajusta el área mínima según sea necesario
                # Contar la lata de Coca-Cola detectada
                latas_detectadas.add(lata_id)
                print("¡Lata de Coca-Cola detectada!")

# Función para procesar la detección en el dron Tello
def procesar_deteccion_tello():
    global latas_detectadas

    # Conectar con el dron Tello
    tello = Tello()
    tello.connect()
    tello.streamon()  # Iniciar la transmisión de video

    # Ciclo para procesar la detección en tiempo real
    while True:
        # Capturar un frame de la transmisión de video del dron Tello
        frame = tello.get_frame_read().frame

        # Realizar inferencia con el modelo de Roboflow
        response = model.predict(frame, confidence=40, overlap=30).json()

        # Procesar los resultados de la detección
        procesar_deteccion(response)

        # Mostrar el frame con las detecciones
        cv2.imshow("Coca-Cola Detection", frame)

        # Esperar y verificar si se presiona la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Detener la transmisión de video y cerrar la ventana
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()

# Función principal
def main():
    choice = input("Selecciona la cámara (tello/windows): ").lower()

    if choice == "tello":
        procesar_deteccion_tello()  # Procesar la detección en el dron Tello
    elif choice == "windows":
        print("Esta opción no está implementada todavía.")  # Aquí puedes agregar la lógica para la cámara de Windows
    else:
        print("Opción no válida. Selecciona 'tello' o 'windows'.")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
