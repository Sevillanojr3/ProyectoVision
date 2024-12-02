import cv2
import numpy as np
import mediapipe as mp
import os

# Deshabilitar transformaciones de hardware para evitar errores
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detectar_gesto(contorno, hull, area):
    # Definir un umbral de área para distinguir entre mano abierta y cerrada
    AREA_UMBRAL = 39000  # Ajustar este valor según sea necesario
    print(f"Área actual: {area}")
    
    if area > AREA_UMBRAL:
        return "Mano abierta"
    else:
        return "Mano cerrada"


def main():
    try:
        cap = cv2.VideoCapture(1)  # Cambia el índice si es necesario
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara. Intentando con índice 0...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: No se pudo abrir ninguna cámara")
                return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Crear una copia del frame para dibujar
            output_frame = frame.copy()
            # Crear máscara negra del tamaño del frame
            mascara_negra = np.zeros_like(frame)

            gesture = "Gesto no detectado"
            ok_gesture = False
            pos_x = 0
            pos_y = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape

                    # Obtener las coordenadas de la mano sin usar los landmarks
                    x_list = []
                    y_list = []
                    for lm in hand_landmarks.landmark:
                        x_list.append(int(lm.x * w))
                        y_list.append(int(lm.y * h))

                    # Calcular el centro de la mano relativo al centro del frame
                    pos_x = (sum(x_list) // len(x_list)) - (w // 2)
                    pos_y = (sum(y_list) // len(y_list)) - (h // 2)

                    # Crear una ROI alrededor de la mano detectada con margen adicional
                    x_min = max(0, min(x_list) - 40)  # Aumentado el margen
                    y_min = max(0, min(y_list) - 40)
                    x_max = min(w, max(x_list) + 40)
                    y_max = min(h, max(y_list) + 40)

                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue

                    # Mejorar el preprocesamiento de la imagen
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Aplicar operaciones morfológicas para mejorar el contorno
                    kernel = np.ones((5,5), np.uint8)
                    thresh = cv2.dilate(thresh, kernel, iterations=1)
                    thresh = cv2.erode(thresh, kernel, iterations=1)

                    # Encontrar contornos en la imagen binaria
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # Seleccionar el contorno más grande
                        contorno_mano = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(contorno_mano)

                        if area > 3000:
                            # Mejorar la aproximación del contorno
                            epsilon = 0.001 * cv2.arcLength(contorno_mano, True)
                            approx = cv2.approxPolyDP(contorno_mano, epsilon, True)

                            # Encontrar el hull y los defectos de convexidad
                            hull_points = cv2.convexHull(approx)
                            hull_indices = cv2.convexHull(approx, returnPoints=False)
                            defects = cv2.convexityDefects(approx, hull_indices) if len(hull_indices) > 3 else None

                            # Dibujar contornos en el ROI con colores más visibles
                            cv2.drawContours(roi, [approx], -1, (0, 255, 0), 3)
                            cv2.drawContours(roi, [hull_points], -1, (0, 0, 255), 3)

                            # Detectar gestos
                            gesture = detectar_gesto(approx, hull_points, area)

                    # Copiar ROI al frame de salida y a la máscara negra
                    mascara_negra[y_min:y_max, x_min:x_max] = roi
                    output_frame = mascara_negra.copy()

                    # Dibujar rectángulo alrededor de la mano
                    cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    # Dibujar punto central de la mano
                    cv2.circle(output_frame, (w//2 + pos_x, h//2 + pos_y), 5, (0, 255, 255), -1)

            # Mostrar el gesto detectado y la posición
            cv2.putText(output_frame, f'Gesto: {gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output_frame, f'Pos: ({pos_x}, {pos_y})', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Dibujar líneas de referencia en el centro
            h, w, _ = output_frame.shape
            cv2.line(output_frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
            cv2.line(output_frame, (0, h//2), (w, h//2), (255, 255, 255), 1)

            cv2.imshow('Detector de Gestos', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()