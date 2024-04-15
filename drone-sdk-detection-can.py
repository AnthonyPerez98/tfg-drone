from inference_sdk import InferenceHTTPClient
from djitellopy import Tello

# Inicializar el cliente de inferencia de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="tVYeBTfHjApBrSJzNiQF"
)

# Función para realizar la detección de objetos con el dron Tello
def detect_objects_with_tello():
    # Conectar al dron Tello
    tello = Tello()
    tello.connect()

    # Configurar la resolución de la cámara
    tello.set_video_resolution('Tello.RESOLUTION_480P')

    # Iniciar el flujo de video
    tello.streamon()

    try:
        # Realizar la detección de objetos mientras se recibe el flujo de video
        while True:
            # Capturar un fotograma del flujo de video
            frame = tello.get_frame_read().frame

            # Enviar el fotograma para la detección de objetos
            result = CLIENT.infer(frame, model_id="tfg-ls1lh/2")

            # Procesar el resultado de la detección
            process_detection(result)

    except KeyboardInterrupt:
        # Detener el flujo de video y cerrar la conexión con el dron Tello al presionar Ctrl+C
        tello.streamoff()
        tello.end()

# Función para procesar el resultado de la detección de objetos
def process_detection(result):
    # Procesar el resultado aquí, por ejemplo, imprimir las predicciones
    print(result)

# Función principal para iniciar la detección de objetos con el dron Tello d
def main():
    detect_objects_with_tello()

if __name__ == "__main__":
    main()
