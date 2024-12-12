import cv2
import numpy as np
import mediapipe as mp
import os
import traceback # Añadido para mejor manejo de errores

# Deshabilitar transformaciones de hardware para evitar errores
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"

# Inicializar MediaPipe Hands para ambas cámaras
mp_hands = mp.solutions.hands
hands1 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
hands2 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detectar_gesto(contorno, area):
    # Definir un umbral de área para distinguir entre mano abierta y cerrada
    AREA_UMBRAL = 15000  # Ajustar este valor según sea necesario

    if area > AREA_UMBRAL:
        return "Mano abierta"
    else:
        return "Mano cerrada"

def detectar_dislike(contorno):
    try:
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contorno)

        if hull_area == 0:
            return False

        solidity = float(contour_area) / hull_area

        # Ajustamos los parámetros para una mejor detección del gesto de paz
        if 0.6 <= solidity <= 0.85:  # Rango ajustado para el gesto de paz
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h

            # Ajustamos los criterios para el gesto de paz
            if 6 <= len(approx) <= 7 and 0.70 <= aspect_ratio <= 0.90:
                return "Dislike"
        return "Gesto no reconocido"
    except:
        return "Gesto no reconocido"

def detectar_dino(contorno):
    try:
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contorno)

        if hull_area == 0:
            return False

        solidity = float(contour_area) / hull_area
        print("solidity", solidity)

        # Ajustamos los parámetros para una mejor detección del gesto de paz
        if 0.6 <= solidity <= 0.85:  # Rango ajustado para el gesto de paz
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h

            # Ajustamos los criterios para el gesto de paz
            if 8 <= len(approx) <= 9 and 1.0 <= aspect_ratio <= 1.46:
                return "Dino"
        return "Gesto no reconocido"
    except:
        return "Gesto no reconocido"

def detectar_paz(contorno):
    try:
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contorno)

        if hull_area == 0:
            return False

        solidity = float(contour_area) / hull_area

        # Ajustamos los parámetros para una mejor detección del gesto de paz
        if 0.6 <= solidity <= 0.85:  # Rango ajustado para el gesto de paz
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h

            # Ajustamos los criterios para el gesto de paz
            if 7 <= len(approx) <= 8 and 0.40 <= aspect_ratio <= 60:
                return "Paz"
        return "Gesto no reconocido"
    except:
        return "Gesto no reconocido"

def main():
    try:
        print("Iniciando programa...")
        cap1 = cv2.VideoCapture(0)  # Primera cámara
        cap2 = cv2.VideoCapture(1)  # Segunda cámara
        
        if not cap1.isOpened():
            print("Error: No se pudo abrir la cámara 1. Intentando con índice 0...")
            cap1 = cv2.VideoCapture(1)
            if not cap1.isOpened():
                print("Error: No se pudo abrir la cámara 1")
                return

        if not cap2.isOpened():
            print("Error: No se pudo abrir la cámara 2. Intentando con siguiente índice...")
            cap2 = cv2.VideoCapture(2)
            if not cap2.isOpened():
                print("Error: No se pudo abrir la cámara 2")
                return

        fps = 10  # Reducir a 10 fps
        size = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Resolución de cámara: {size}")

        while True:
            try:
                # Inicializar todas las variables de gestos al principio del bucle
                gesture_basic = "Gesto no detectado"
                gesture_paz = "Gesto no reconocido"
                gesture_dino = "Gesto no reconocido"
                gesture_dislike = "Gesto no reconocido"
                pos_x = 0
                pos_y = 0
                pos_z = 0
                pos_x2 = 0

                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1: #or not ret2:
                    print("Error al leer los frames")
                    break

                frame1 = cv2.flip(frame1, 1)
                frame2 = cv2.flip(frame2, 1)
                
                rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                
                results1 = hands1.process(rgb_frame1)
                results2 = hands2.process(rgb_frame2)

                output_frame1 = frame1.copy()
                output_frame2 = frame2.copy()
                mascara_negra1 = np.zeros_like(frame1)
                mascara_negra2 = np.zeros_like(frame2)

                # Procesar cámara 1
                if results1.multi_hand_landmarks:
                    for hand_landmarks in results1.multi_hand_landmarks:
                        h, w, _ = frame1.shape

                        x_list = []
                        y_list = []
                        for lm in hand_landmarks.landmark:
                            x_list.append(int(lm.x * w))
                            y_list.append(int(lm.y * h))

                        pos_x = (sum(x_list) // len(x_list)) - (w // 2)
                        pos_y = (sum(y_list) // len(y_list)) - (h // 2)

                        x_min = max(0, min(x_list) - 40)
                        y_min = max(0, min(y_list) - 40)
                        x_max = min(w, max(x_list) + 40)
                        y_max = min(h, max(y_list) + 40)

                        roi = frame1[y_min:y_max, x_min:x_max]
                        if roi.size == 0:
                            continue

                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresh = cv2.bitwise_not(thresh)

                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            contorno_mano = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(contorno_mano)

                            if area > 3000:
                                cv2.drawContours(roi, [contorno_mano], -1, (0, 255, 0), 3)
                                gesture_basic = detectar_gesto(contorno_mano,area)
                                gesture_paz = detectar_paz(contorno_mano)
                                gesture_dino = detectar_dino(contorno_mano)
                                gesture_dislike = detectar_dislike(contorno_mano)

                        mascara_negra1[y_min:y_max, x_min:x_max] = roi
                        output_frame1 = mascara_negra1.copy()
                        cv2.rectangle(output_frame1, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.circle(output_frame1, (w//2 + pos_x, h//2 + pos_y), 5, (0, 255, 255), -1)

                # Procesar cámara 2 (usando pos_y como pos_z)
                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        h, w, _ = frame2.shape

                        x_list = []
                        y_list = []
                        for lm in hand_landmarks.landmark:
                            x_list.append(int(lm.x * w))
                            y_list.append(int(lm.y * h))

                        pos_x2 = (sum(x_list) // len(x_list)) - (w // 2)
                        pos_z = (sum(y_list) // len(y_list)) - (h // 2)

                        x_min = max(0, min(x_list) - 40)
                        y_min = max(0, min(y_list) - 40)
                        x_max = min(w, max(x_list) + 40)
                        y_max = min(h, max(y_list) + 40)

                        roi = frame2[y_min:y_max, x_min:x_max]
                        if roi.size == 0:
                            continue

                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresh = cv2.bitwise_not(thresh)

                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            contorno_mano = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(contorno_mano)

                            if area > 3000:
                                cv2.drawContours(roi, [contorno_mano], -1, (0, 255, 0), 3)

                        mascara_negra2[y_min:y_max, x_min:x_max] = roi
                        output_frame2 = mascara_negra2.copy()
                        cv2.rectangle(output_frame2, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.circle(output_frame2, (w//2 + pos_x2, h//2 + pos_z), 5, (0, 255, 255), -1)

                # Mostrar información en las ventanas
                cv2.putText(output_frame1, f'Gesto: {gesture_basic}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame1, f'Paz: {gesture_paz}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame1, f'Dino: {gesture_dino}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame1, f'Dislike: {gesture_dislike}', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame1, f'Pos: ({pos_x}, {pos_y})', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(output_frame2, f'Pos: ({pos_x2}, {pos_z})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Dibujar líneas de referencia
                h, w, _ = output_frame1.shape
                cv2.line(output_frame1, (w//2, 0), (w//2, h), (255, 255, 255), 1)
                cv2.line(output_frame1, (0, h//2), (w, h//2), (255, 255, 255), 1)
                cv2.line(output_frame2, (w//2, 0), (w//2, h), (255, 255, 255), 1)
                cv2.line(output_frame2, (0, h//2), (w, h//2), (255, 255, 255), 1)

                cv2.imshow('Cámara 1 - Gestos', output_frame1)
                cv2.imshow('Cámara 2 - Profundidad', output_frame2)

                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    print("Programa terminado por el usuario")
                    break

            except Exception as e:
                print(f"Error en el bucle principal: {str(e)}")
                print("Traza completa del error:")
                print(traceback.format_exc())
                continue

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        print("Traza completa del error:")
        print(traceback.format_exc())
    
    finally:
        print("Cerrando programa...")
        if 'cap1' in locals():
            cap1.release()
        if 'cap2' in locals():
            cap2.release()
        cv2.destroyAllWindows()
        print("Programa finalizado")

if __name__ == "__main__":
    main()
