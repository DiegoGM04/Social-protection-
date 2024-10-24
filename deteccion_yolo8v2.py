import cv2
import torch
import numpy as np
import easyocr  # OCR para detectar placas
from ultralytics import YOLO

source_yolo8 = "C:\\Users\\js023\\Downloads\\yolo\\yolo\\yolov8m.pt"
# Cargar el modelo YOLOv8 desde un archivo local
model = YOLO(source_yolo8)  

# Inicializar el OCR de easyocr
reader = easyocr.Reader(['en'])

# Parametros
history_length = 5
object_history = {}
collision_messages = []

# Funcion para cargar y configurar la camara
def initialize_camera(camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se pudo acceder a la camara.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

# Funcion para verificar colision entre dos cuadros (bounding boxes)
def is_collision(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

# Funcion para detectar objetos en el frame
def detect_objects(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    detected_objects = []
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:  # Comprobar si hay detecciones
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas
                w, h = x2 - x1, y2 - y1
                
                # Reducir los tamaños para las colisiones
                w = int(w * 0.75)  # Reducir el ancho en un 25%
                h = int(h * 0.75)  # Reducir la altura en un 25%

                label = model.names[int(box.cls)]  # Obtener el nombre del objeto

                if label in ["person", "car", "truck", "motorcycle"]:
                    detected_objects.append((label, (x1, y1, w, h)))
                    update_object_history(label, (x1, y1, w, h))

    return detected_objects

# Funcion para actualizar el historial de posiciones de objetos
def update_object_history(label, box):
    if label not in object_history:
        object_history[label] = []
    object_history[label].append(box)
    if len(object_history[label]) > history_length:
        object_history[label].pop(0)

# Funcion para detectar placas y aplicar OCR
def detect_plate(frame, box):
    x1, y1, w, h = box
    plate_region = frame[y1 + int(h * 0.7):y1 + h, x1:x1 + w]
    if plate_region.size > 0:
        result = reader.readtext(plate_region)
        if result:
            plate_text = result[0][-2]
            save_plate_to_txt(plate_text)
            return plate_text
    return None

# Funcion para guardar el numero de placa en un archivo de texto
def save_plate_to_txt(plate_number):
    with open("placas_detectadas.txt", "a") as file:
        file.write(f"{plate_number}\n")

# Funcion para verificar colisiones
def check_collisions(detected_objects):
    global collision_messages
    collision_messages = []

    for i, obj1 in enumerate(detected_objects):
        label1, (x1, y1, w1, h1) = obj1
        for j, obj2 in enumerate(detected_objects):
            if i != j:
                label2, (x2, y2, w2, h2) = obj2

                # Verificar colisiones entre vehiculos y personas, y motocicletas
                if (label1 in ["car", "truck", "motorcycle"] and label2 in ["car", "truck", "motorcycle"]) or \
                   (label1 in ["car", "truck", "motorcycle"] and label2 == "person") or \
                   (label1 == "person" and label2 in ["car", "truck", "motorcycle"]):
                    if is_collision((x1, y1, w1, h1), (x2, y2, w2, h2)):
                        collision_messages.append(f"Colisión detectada entre {label1} y {label2}")

# Funcion para dibujar cuadros de deteccion
def draw_detections(frame, detected_objects):
    for obj in detected_objects:
        label, (x1, y1, w, h) = obj

        # Verificar si la persona está tumbada
        if label == "person":
            aspect_ratio = w / h
            color = (0, 0, 255) if aspect_ratio > 1.5 else (0, 255, 0)  # Rojo si está tumbada, verde si está de pie
        else:
            color = (255, 0, 0)  # Azul para otros objetos (carros, camiones, motos)

        # Dibujar el cuadro
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detectar y mostrar la placa si es un auto
        if label == "car":
            plate_text = detect_plate(frame, (x1, y1, w, h))
            if plate_text:
                cv2.putText(frame, f"Placa: {plate_text}", (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Funcion principal para procesar el video
def process_video():
    cap = initialize_camera(1)
    if cap is None:
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        detected_objects = detect_objects(frame)
        draw_detections(frame, detected_objects)
        check_collisions(detected_objects)

        # Mostrar mensajes de colision
        collision_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for idx, message in enumerate(collision_messages):
            cv2.putText(collision_frame, message, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Deteccion de Colisiones y Objetos", frame)
        cv2.imshow("Imagen Original", original_frame)
        cv2.imshow("Mensajes de Colision", collision_frame)

        # Salir si se presiona la tecla q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar el programa
if __name__ == "__main__":
    process_video()
