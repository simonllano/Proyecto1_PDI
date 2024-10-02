import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from game import (
    load_background,
    load_turtles,
    initialize_camera,
    get_frame,
    create_mask,
    clean_mask,
    binarize_mask,
    draw_circle_and_turtle,
    overlay_medusa,
    GameObject,
    Enemy 
)

def main():
    desired_size = (960, 540)
    background = load_background('sea (1).jpg', desired_size)
    turtle_left, turtle_right = load_turtles('turtle_left.png', 'turtle_right.png')
    enemy_image = cv2.imread('basura.png', cv2.IMREAD_UNCHANGED)  # Carga con canal alfa
    medusa=cv2.imread('medusa.png', cv2.IMREAD_UNCHANGED)
   

    cap = initialize_camera(960, 560)
    kernel = np.ones((10, 10), np.uint8)  # Definir el kernel aquí

    objects = []
    for _ in range(10):  # Crear 6 medusas
        pos = [np.random.randint(0, desired_size[0] - 50), np.random.randint(0, desired_size[1] - 50)]
        speed = [np.random.choice([-2, 2]), np.random.choice([-2, 2])]
        objects.append(GameObject(medusa, pos, speed))

    enemies = []
    for _ in range(10):  # Crear 12 enemigos estáticos
        pos = [np.random.randint(0, desired_size[0] - 50), np.random.randint(0, desired_size[1] - 50)]
        enemies.append(Enemy(enemy_image, pos))

    cv2.namedWindow('Tapa con Fondo y Tortuga')
    
    score = 0 #Inicializar puntaje
    lives = 3
    hit_flag = False
    last_hit_time = 0  # Variable para almacenar el tiempo del último golpe
    hit_cooldown = 0.5  # Tiempo en segundos para el cooldown
    
    while True:
        frame = get_frame(cap)
        if frame is None:
            print("No se pudo capturar la imagen de la cámara")
            break

        frame_resized = cv2.resize(frame, desired_size)
        ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)  # Cambiar a YCrCb

        mask = create_mask(ycrcb)  # Crear máscara en el espacio YCrCb
        cleaned_mask = clean_mask(mask, kernel)
        binary_mask = binarize_mask(cleaned_mask)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        combined = background.copy()
        
        center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))

        # Mover y dibujar las medusas
        for obj in objects:
            obj.move(desired_size, turtle_pos=center)  # Pasar la posición de la tortuga
            combined = overlay_medusa(combined, obj)
            # Verificar si la medusa ha sido capturada
            if center is not None:  # Asegurarse de que center está definido
                if not obj.caught and  obj.is_caught(center, radius):  # Comprobar captura
                    #print("Medusa capturada!")
                    obj.caught = True  # Marcar la medusa como capturada
                    score += 1 
                
        # Verificar si todas las medusas han sido capturadas
        all_caught = all(obj.caught for obj in objects)  # Verifica si todas las medusas están capturadas
        if all_caught:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, 'Ganaste!!', (desired_size[0] // 3, desired_size[1] // 2), 
                        font, 2, (0, 255, 0), 3, cv2.LINE_AA)  
            cv2.imshow('Tapa con Fondo y Tortuga', combined)
            cv2.waitKey(2000)  # Esperar 3 segundos para que el mensaje sea visible
            break 
                    
        # Dibujar enemigos y comprobar colisiones
        for enemy in enemies:
            enemy.draw(combined)
            # Verificar colisión con la tortuga
            if contours:
                for c in contours:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    if enemy.pos[0] <= center[0] <= enemy.pos[0] + enemy.size[0] and \
                       enemy.pos[1] <= center[1] <= enemy.pos[1] + enemy.size[1]:
                        #print("¡Has chocado con un enemigo! ¡Perdiste!")
                        
                        if not hit_flag:  # Solo si no ha sido golpeado recientemente
                            hit_flag = True  # Marcar que ha sido golpeado
                            last_hit_time = time.time()  # Guardar el tiempo del golpe
                            lives -= 1  # Restar una vida
                            if lives <= 0:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(combined, 'Perdiste!!', (desired_size[0] // 3, desired_size[1] // 2), 
                                            font, 2, (0, 0, 255), 3, cv2.LINE_AA)  # Mostrar 'Perdiste' en rojo
                                cv2.imshow('Tapa con Fondo y Tortuga', combined)
                                cv2.waitKey(2000)  # Esperar 3 segundos antes de salir
                                cap.release()
                                cv2.destroyAllWindows()
                                break  # Termina el juego
                                
                    # Verificar si ha pasado el cooldown desde el último golpe
                    if hit_flag and (time.time() - last_hit_time >= hit_cooldown):
                        hit_flag = False  # Restablecer la bandera después del tiempo de cooldown
                        
                          

        # Dibuja la tortuga
        combined = draw_circle_and_turtle(combined, contours, turtle_left, turtle_right, desired_size)

        # Mostrar el puntaje en la pantalla
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Puntaje: {score}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Mostrar vidas restantes
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'Vidas: {lives}', (800, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Camara', frame)

        cv2.imshow('Tapa con Fondo y Tortuga', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
