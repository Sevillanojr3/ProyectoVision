import cv2
import numpy as np
import mediapipe as mp
import os
import traceback # Añadido para mejor manejo de errores

# Deshabilitar transformaciones de hardware para evitar errores
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detectar_gesto(contorno, area):
    # Definir un umbral de área para distinguir entre mano abierta y cerrada
    AREA_UMBRAL = 39000  # Ajustar este valor según sea necesario
    #print(f"Área actual: {area}")
    
    if area > AREA_UMBRAL:
        return "Mano abierta"
    else:
        return "Mano cerrada"

def detectar_gestos_avanzados(contorno):
    try:
        # Calcular el casco convexo y su área
        hull = cv2.convexHull(contorno)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contorno)

        if hull_area == 0:
            return "Gesto no reconocido"

        # Calcular la solidez
        solidity = float(contour_area) / hull_area

        # Aproximar el contorno para reducir puntos
        epsilon = 0.02 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)

        # Calcular el aspecto de la caja delimitadora
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = float(w) / h

        # Calcular la extensión (proporción del área del contorno al área del rectángulo)
        rect_area = w * h
        if rect_area == 0:
            return "Gesto no reconocido"
        extent = float(contour_area) / rect_area
        if len(approx) <= 6 and aspect_ratio < 0.5 and solidity < 0.85:
            return "Pulgar a un lado"
        elif len(approx) >= 8 and aspect_ratio > 0.3 and aspect_ratio < 0.7:
            return "Señal de paz"
        else:
            return "Gesto no reconocido"
    except Exception as e:
        print(f"Error en detectar_gestos_avanzados: {str(e)}")
        return "Error en detección"

def main():
    try:
        print("Iniciando programa...")
        # Inicializar ambas cámaras
        cap1 = cv2.VideoCapture(1)  # Cámara principal
        cap2 = cv2.VideoCapture(2)  # Cámara para eje Z
        
        # Verificar primera cámara
        if not cap1.isOpened():
            print("Error: No se pudo abrir la cámara 1. Intentando con índice 0...")
            cap1 = cv2.VideoCapture(0)
            
            if not cap1.isOpened():
                print("Error: No se pudo abrir la cámara principal")
                return

        # Verificar segunda cámara
        if not cap2.isOpened():
            print("Error: No se pudo abrir la cámara 2. Intentando con índice 3...")
            cap2 = cv2.VideoCapture(3)
            
            if not cap2.isOpened():
                print("Error: No se pudo abrir la cámara secundaria")
                return

        fps = 10  # Reducir a 10 fps
        size = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Resolución de cámara principal: {size}")

        while True:  # Bucle infinito, solo se cierra con 'q'
            try:
                # Leer frames de ambas cámaras
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Error al leer los frames")
                    break

                frame1 = cv2.flip(frame1, 1)
                frame2 = cv2.flip(frame2, 1)
                
                rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                
                results1 = hands.process(rgb_frame1)
                results2 = hands.process(rgb_frame2)

                # Crear copias de los frames para dibujar
                output_frame1 = frame1.copy()
                output_frame2 = frame2.copy()
                
                # Crear máscaras negras
                mascara_negra1 = np.zeros_like(frame1)
                mascara_negra2 = np.zeros_like(frame2)

                gesture_basic = "Gesto no detectado"
                gesture_advanced = "Gesto no detectado"
                pos_x = 0
                pos_y = 0
                pos_z = 0  # Nueva variable para profundidad

                # Procesar cámara principal (XY)
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
                                gesture_basic = detectar_gesto(contorno_mano, area)
                                gesture_advanced = detectar_gestos_avanzados(contorno_mano)

                        mascara_negra1[y_min:y_max, x_min:x_max] = roi
                        output_frame1 = mascara_negra1.copy()
                        cv2.rectangle(output_frame1, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.circle(output_frame1, (w//2 + pos_x, h//2 + pos_y), 5, (0, 255, 255), -1)

                # Procesar cámara secundaria (Z)
                if results2.multi_hand_landmarks:
                    for hand_landmarks in results2.multi_hand_landmarks:
                        h, w, _ = frame2.shape
                        y_list = []
                        for lm in hand_landmarks.landmark:
                            y_list.append(int(lm.y * h))
                            
                        # Calcular posición Z usando la posición Y de la segunda cámara
                        pos_z = (sum(y_list) // len(y_list)) - (h // 2)
                        
                        # Obtener ROI de la mano en la segunda cámara
                        y_min = max(0, min(y_list) - 40)
                        y_max = min(h, max(y_list) + 40)
                        x_min = 0
                        x_max = w
                        
                        roi2 = frame2[y_min:y_max, x_min:x_max]
                        if roi2.size > 0:
                            # Procesar ROI similar a la primera cámara
                            gray_roi = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            thresh = cv2.bitwise_not(thresh)
                            
                            mascara_negra2[y_min:y_max, x_min:x_max] = roi2
                            output_frame2 = mascara_negra2.copy()
                            cv2.rectangle(output_frame2, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.circle(output_frame2, (w//2, h//2 + pos_z), 5, (0, 255, 255), -1)
                            
                            # Mostrar profundidad en la segunda ventana
                            cv2.putText(output_frame2, f'Profundidad: {pos_z}', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Mostrar información en la ventana principal
                cv2.putText(output_frame1, f'Gesto: {gesture_basic} {gesture_advanced}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame1, f'Pos: ({pos_x}, {pos_y}, {pos_z})', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Dibujar líneas de referencia
                h, w, _ = output_frame1.shape
                cv2.line(output_frame1, (w//2, 0), (w//2, h), (255, 255, 255), 1)
                cv2.line(output_frame1, (0, h//2), (w, h//2), (255, 255, 255), 1)

                # Mostrar ambas ventanas
                cv2.imshow('Detector de Gestos XY', output_frame1)
                cv2.imshow('Detector de Profundidad Z', output_frame2)

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
