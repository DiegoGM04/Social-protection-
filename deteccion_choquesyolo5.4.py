import cv2
import torch
import numpy as np

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Iniciar la captura de video
cap = cv2.VideoCapture(1)

# Verificar si la cámara está abierta
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Configurar la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Almacenar los objetos detectados y sus posiciones anteriores
detected_objects = []  # Lista para almacenar objetos detectados
object_history = {}  # Diccionario para almacenar la historia de posiciones
collision_messages = []  # Lista para almacenar los mensajes de colisión

# Parámetros
history_length = 5  # Cuántos fotogramas guardar en la historia

def is_collision(box1, box2):
    """Función para verificar si hay colisión entre dos objetos"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Determinar si los cuadros se intersectan
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

while cap.isOpened():
    # Leer el siguiente cuadro
    ret, frame = cap.read()
    if not ret:
        break

    # Clonar el frame original para mostrarlo sin detección
    original_frame = frame.copy()

    # Convertir el frame de BGR (OpenCV) a RGB (YOLOv5)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la detección con YOLOv5
    results = model(img)

    # Extraer resultados
    labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]
    
    detected_objects.clear()  # Limpiar lista de objetos detectados

    for i in range(len(labels)):
        confidence = confidences[i]
        if confidence > 0.5:  # Detección con confianza mayor al 50%
            # Obtener las coordenadas del bounding box
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            w, h = x2 - x1, y2 - y1

            # Obtener la clase detectada
            label = model.names[int(labels[i])]

            # Solo incluir autos, camiones, motos y personas
            if label in ["person", "car", "truck", "motorbike"]:
                detected_objects.append((label, (x1, y1, w, h)))

                # Guardar la posición en la historia
                if label not in object_history:
                    object_history[label] = []
                object_history[label].append((x1, y1, w, h))

                # Limitar la longitud de la historia
                if len(object_history[label]) > history_length:
                    object_history[label].pop(0)

    # Detectar colisiones utilizando la historia de posiciones
    collision_messages.clear()  # Limpiar mensajes de colisión
    for i, obj1 in enumerate(detected_objects):
        label1, (x1, y1, w1, h1) = obj1
        for j, obj2 in enumerate(detected_objects):
            if i != j:
                label2, (x2, y2, w2, h2) = obj2
                
                # Ignorar colisiones entre personas
                if label1 == "person" and label2 == "person":
                    continue

                # Verificar si hay colisión
                if is_collision((x1, y1, w1, h1), (x2, y2, w2, h2)):
                    collision_messages.append(f"Colisión detectada entre {label1} y {label2}")

                # Comprobar la historia de posiciones para prever colisiones
                if label1 in object_history and label2 in object_history:
                    last_pos1 = object_history[label1][-1]  # Última posición de obj1
                    last_pos2 = object_history[label2][-1]  # Última posición de obj2

                    # Calcular la dirección del movimiento
                    dir1 = (x1 - last_pos1[0], y1 - last_pos1[1])
                    dir2 = (x2 - last_pos2[0], y2 - last_pos2[1])

                    # Predecir las posiciones futuras
                    future_pos1 = (x1 + dir1[0], y1 + dir1[1])
                    future_pos2 = (x2 + dir2[0], y2 + dir2[1])

                    # Verificar colisión en posiciones futuras
                    if is_collision((future_pos1[0], future_pos1[1], w1, h1), (future_pos2[0], future_pos2[1], w2, h2)):
                        collision_messages.append(f"Colisión prevista entre {label1} y {label2} en el futuro.")

    # Dibujar los cuadros de detección
    for obj in detected_objects:
        label, (x1, y1, w, h) = obj

        # Verificar si la persona está tumbada
        if label == "person":
            aspect_ratio = w / h
            if aspect_ratio > 1.5:  # Si el ancho es mucho mayor que la altura
                color = (0, 0, 255)  # Rojo si la persona está tumbada
            else:
                color = (0, 255, 0)  # Verde si la persona está de pie
        else:
            color = (255, 0, 0)  # Azul para otros objetos

        # Dibujar el cuadro
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Crear un frame para mostrar mensajes de colisión
    collision_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Crear un frame negro

    # Escribir los mensajes de colisión en el frame
    for idx, message in enumerate(collision_messages):
        cv2.putText(collision_frame, message, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar el cuadro con la detección
    cv2.imshow("Detección de Colisiones y Objetos", frame)

    # Mostrar el frame original sin detección de objetos
    cv2.imshow("Imagen Original", original_frame)

    # Mostrar el frame de mensajes de colisión
    cv2.imshow("Mensajes de Colisión", collision_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
