from inference_sdk import InferenceHTTPClient
from djitellopy import Tello
import cv2

# Inicializar el cliente de inferencia de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="PUT_YOUR_API_KEY"
)


# Función para capturar un fotograma de la cámara Tello
def capture_tello_frame(tello):
    frame = tello.get_frame_read().frame
    return frame


# Función para procesar el resultado de la detección de objetos
def process_detection(result, frame):
    # Procesar el resultado aquí, por ejemplo, imprimir las predicciones
    print(result)

    # Dibujar el cuadro delimitador y etiqueta en el fotograma
    if 'predictions' in result:
        for pred in result['predictions']:
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            width = int(pred['width'])
            height = int(pred['height'])
            class_label = pred['class']

            # Dibujar el cuadro delimitador
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            # Dibujar la etiqueta de clase
            cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow("Tello Detection", frame)


# Función para realizar la detección de objetos con el dron Tello
def detect_objects_with_tello():
    # Conectar al dron Tello
    tello = Tello()
    tello.connect()

    # Iniciar el flujo de video
    tello.streamon()

    try:
        # Realizar la detección de objetos mientras se recibe el flujo de video
        while True:
            # Capturar un fotograma del flujo de video
            frame = capture_tello_frame(tello)

            # Redimensionar el fotograma a una resolución menor para mejorar la fluidez
            frame_resized = cv2.resize(frame, (640, 480))

            # Convertir el fotograma a formato RGB (Roboflow puede requerir este formato)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Enviar el fotograma para la detección de objetos
            result = CLIENT.infer(frame_rgb, model_id="tfg-ls1lh/2")

            # Procesar el resultado de la detección
            process_detection(result, frame_resized)

            # Verificar si se presiona la tecla 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Detener el flujo de video y cerrar la conexión con el dron Tello
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


# Función principal para iniciar la detección de objetos con el dron Tello
def main():
    detect_objects_with_tello()


if __name__ == "__main__":
    main()
