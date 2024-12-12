#!/usr/bin/env python3

import rospy
from proyecto_final.msg import HandData
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp 
import numpy as np

def detectar_gesto(contorno, area):
    AREA_UMBRAL = 39000
    if area > AREA_UMBRAL:
        return True
    return False

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
            print("approx", len(approx))

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            print("aspect_ratio", aspect_ratio)

            # Ajustamos los criterios para el gesto de paz
            if len(approx) >= 11 and 0.55 <= aspect_ratio <= 0.8:
                return True
        return False
    except:
        return False

def detectar_dino(contorno):
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
            print("approx", len(approx))

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            print("aspect_ratio", aspect_ratio)

            # Ajustamos los criterios para el gesto de paz
            if len(approx) >= 9 and 1.0 <= aspect_ratio <= 1.46:
                return True
        return False
    except:
        return False

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
            #print("approx", len(approx))

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            #print("aspect_ratio", aspect_ratio)

            # Ajustamos los criterios para el gesto de paz
            if len(approx) >= 8 and  len(approx) <= 10 and 0.4 <= aspect_ratio <= 0.8:
                return True
        return False
    except:
        return False

def hand_detector():
    rospy.init_node('hand_detector_node', anonymous=True)
    pub = rospy.Publisher('/hand_data', HandData, queue_size=10)

    mp_hands = mp.solutions.hands
    hands1 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    # hands2 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    cap1 = cv2.VideoCapture(0)
    # cap2 = cv2.VideoCapture(1)

    rate = rospy.Rate(10)

    window_name = "Cámara 1"
    cv2.namedWindow(window_name)

    while not rospy.is_shutdown():
        ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()

        if not ret1:
            rospy.logwarn("No se pudieron capturar frames de las cámaras")
            continue

        frame1 = cv2.flip(frame1, 1)
        # frame2 = cv2.flip(frame2, 1)

        rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        results1 = hands1.process(rgb_frame1)
        # results2 = hands2.process(rgb_frame2)

        # Inicializar hand_data con valores por defecto y mano no detectada
        hand_data = HandData()
        hand_data.x = 0.0
        hand_data.y = 0.0
        hand_data.z = 0.0
        hand_data.is_open = True  # Por defecto abierta
        hand_data.is_peace = False
        hand_data.is_dino = False
        hand_data.is_dislike = False

        if results1.multi_hand_landmarks:
            hand_data.hand_detected = True
            # Procesar cámara 1 para x, y
            for hand_landmarks in results1.multi_hand_landmarks:
                h, w, _ = frame1.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                x_min = max(0, min(landmarks, key=lambda p: p[0])[0] - 30)
                y_min = max(0, min(landmarks, key=lambda p: p[1])[1] - 30)
                x_max = min(w, max(landmarks, key=lambda p: p[0])[0] + 30)
                y_max = min(h, max(landmarks, key=lambda p: p[1])[1] + 30)

                centro_x = (x_min + x_max) // 2
                centro_y = (y_min + y_max) // 2
                hand_data.x = float(centro_x - w // 2)
                hand_data.y = float(h // 2 - centro_y)

                # Detección de gestos
                roi = frame1[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(cnt)
                        hand_data.is_open = detectar_gesto(cnt, area)
                        hand_data.is_peace = detectar_paz(cnt)
                        hand_data.is_dino = detectar_dino(cnt)
                        hand_data.is_dislike = detectar_dislike(cnt)

                # Dibujar un círculo verde cuando se detecta la mano
                cv2.circle(frame1, (centro_x, centro_y), 10, (0, 255, 0), -1)
                cv2.putText(frame1, "Mano detectada", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Añadir visualización del estado de la pinza
                estado_texto = "Gesto de paz detectado" if hand_data.is_peace else "Gesto normal"
                cv2.putText(frame1, estado_texto, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if hand_data.is_peace else (0, 0, 255), 2)

                # Añadir visualización del estado de la pinza
                estado_texto = "Gesto de dino detectado" if hand_data.is_dino else "Gesto normal"   
                cv2.putText(frame1, estado_texto, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if hand_data.is_dino else (0, 0, 255), 2)
                # Añadir visualización del estado de la pinza
                estado_texto = "Gesto de dislike" if hand_data.is_dislike else "Gesto normal"   
                cv2.putText(frame1, estado_texto, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if hand_data.is_dislike else (0, 0, 255), 2)
        else:
            # Mostrar texto en rojo cuando no se detecta la mano
            cv2.putText(frame1, "No se detecta mano", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, frame1)
        # cv2.imshow("Cámara 2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pub.publish(hand_data)
        rate.sleep()

    cap1.release()
    # cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        hand_detector()
    except rospy.ROSInterruptException:
        pass