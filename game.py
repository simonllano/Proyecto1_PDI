import cv2
import numpy as np

def load_background(image_path, desired_size):
    background = cv2.imread(image_path)
    return cv2.resize(background, desired_size)

def load_turtles(left_image_path, right_image_path):
    turtle_left = cv2.imread(left_image_path, cv2.IMREAD_UNCHANGED)
    turtle_right = cv2.imread(right_image_path, cv2.IMREAD_UNCHANGED)
    return turtle_left, turtle_right

def initialize_camera(width, height):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    return cv2.flip(frame, 1) if ret else None

def create_mask(ycrcb):
    lower_ycrcb = np.array([0, 163, 29])  # Ajusta los valores según sea necesario
    upper_ycrcb = np.array([255, 230, 225])  # Ajusta los valores según sea necesario
    return cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

def clean_mask(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def binarize_mask(mask):
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return binary_mask

class GameObject:
    def __init__(self, image, pos, speed, escape_threshold=100):
        self.image = image
        self.pos = np.array(pos, dtype=np.float32)
        self.speed = np.array(speed, dtype=np.float32)
        self.size = (image.shape[1], image.shape[0])  # Tamaño del objeto (ancho, alto)
        self.caught = False
        self.escape_threshold = escape_threshold  # Distancia mínima para que la medusa intente escapar

    def move(self, window_size, turtle_pos=None):
        play_area_left = 50
        play_area_top = 50
        play_area_right = window_size[0] - 50
        play_area_bottom = window_size[1] - 50

        if turtle_pos is not None:
            # Calcular la distancia entre la medusa y la tortuga
            distance_to_turtle = np.linalg.norm(self.pos + np.array(self.size) / 2 - turtle_pos)

            if distance_to_turtle < self.escape_threshold:
                # La medusa se mueve en dirección opuesta a la tortuga
                escape_direction = self.pos - turtle_pos
                escape_direction = escape_direction / np.linalg.norm(escape_direction)  # Normalizar
                self.speed = escape_direction * np.linalg.norm(self.speed)  # Mantener la misma velocidad

        self.pos += self.speed

        # Asegurar que la medusa no salga de los límites del área de juego
        if self.pos[0] <= play_area_left or self.pos[0] + self.size[0] >= play_area_right:
            self.speed[0] *= -1
        if self.pos[1] <= play_area_top or self.pos[1] + self.size[1] >= play_area_bottom:
            self.speed[1] *= -1

        self.pos[0] = np.clip(self.pos[0], play_area_left, play_area_right - self.size[0])
        self.pos[1] = np.clip(self.pos[1], play_area_top, play_area_bottom - self.size[1])

    def is_caught(self, center, radius):
        obj_center = self.pos + np.array(self.size) / 2
        distance = np.linalg.norm(obj_center - center)
        if distance <= radius:
            self.caught = True  # Marca como capturada
            return True
        return False

def draw_circle_and_turtle(frame, contours, turtle_left, turtle_right, desired_size, movement_threshold=5):
    combined = frame.copy()
    center = None
    radius = 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))

        cv2.circle(combined, center, int(radius), (0, 255, 0), 2)

        # Detectar movimiento y elegir imagen
        if 'prev_center' in draw_circle_and_turtle.__dict__:
            prev_center = draw_circle_and_turtle.prev_center
            movement = center[0] - prev_center[0]

            if abs(movement) > movement_threshold:
                # Seleccionar tortuga según dirección del movimiento
                draw_circle_and_turtle.turtle_img = turtle_right if movement > 0 else turtle_left

        draw_circle_and_turtle.prev_center = center

    # Si no hay contornos, mantener la posición y dirección de la última tortuga
    if 'prev_center' in draw_circle_and_turtle.__dict__ and 'turtle_img' in draw_circle_and_turtle.__dict__:
        center = draw_circle_and_turtle.prev_center
        turtle_img = draw_circle_and_turtle.turtle_img
    else:
        turtle_img = None

    if center and radius > 0 and turtle_img is not None:
        # Redimensionar la imagen de la tortuga al tamaño del círculo
        turtle_resized = cv2.resize(turtle_img, (int(radius * 2), int(radius * 2)))
        turtle_h, turtle_w, _ = turtle_resized.shape

        top_left_x = center[0] - turtle_w // 2
        top_left_y = center[1] - turtle_h // 2

        if (top_left_x >= 0 and top_left_y >= 0 and
                (top_left_x + turtle_w) <= desired_size[0] and
                (top_left_y + turtle_h) <= desired_size[1]):
            for i in range(turtle_h):
                for j in range(turtle_w):
                    if turtle_resized[i, j][3] != 0:  # Solo si el pixel es no transparente
                        combined[top_left_y + i, top_left_x + j] = turtle_resized[i, j][:3]

    return combined


def overlay_medusa(frame, obj):
    if not obj.caught:  # Solo dibujar si no ha sido capturada
        x, y = obj.pos.astype(int)
        h, w = obj.image.shape[:2]

        if y >= frame.shape[0] or x >= frame.shape[1]:
            return frame  # Si el objeto está fuera de los límites, no lo dibujes

        y_end = min(y + h, frame.shape[0])
        x_end = min(x + w, frame.shape[1])

        # Usar solo el área que se va a superponer
        overlay_region = obj.image[:y_end - y, :x_end - x]

        alpha_channel = overlay_region[:, :, 3] / 255.0  # Canal alfa
        for c in range(3):  # Solo para los canales RGB
            frame[y:y_end, x:x_end, c] = (
                frame[y:y_end, x:x_end, c] * (1 - alpha_channel) +
                overlay_region[:, :, c] * alpha_channel
            )

    return frame


class Enemy:
    def __init__(self, image, pos):
        self.image = image
        self.pos = np.array(pos, dtype=np.float32)
        self.size = (image.shape[1], image.shape[0])  # Tamaño del enemigo (ancho, alto)

    def draw(self, frame):
        x, y = self.pos.astype(int)  # Coordenadas de la posición del enemigo
        h, w = self.image.shape[:2]  # Alto y ancho de la imagen del enemigo

        # Asegurarse de que no se dibuje fuera de los límites de la imagen
        if y >= frame.shape[0] or x >= frame.shape[1]:
            return frame  # Si está fuera de los límites, no dibujes nada

        # Recortar la imagen si está parcialmente fuera de los límites
        overlay_region = self.image

        # Mezcla con el canal alfa si la imagen del enemigo tiene transparencia
        if self.image.shape[2] == 4:  # Si la imagen tiene un canal alfa (imagen RGBA)
            alpha_channel = overlay_region[:, :, 3] / 255.0  # Canal alfa normalizado
            for c in range(3):  # Mezclar los canales RGB
                frame[y:y + h, x:x + w, c] = (
                    frame[y:y + h, x:x + w, c] * (1 - alpha_channel) +
                    overlay_region[:, :, c] * alpha_channel
                )
        else:  # Si no tiene transparencia, simplemente copiar la imagen
            frame[y:y + h, x:x + w] = overlay_region

        return frame


