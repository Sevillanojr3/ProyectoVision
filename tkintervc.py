import cv2
import numpy as np
import mediapipe as mp
import os
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import datetime

# Deshabilitar transformaciones de hardware para evitar errores (ajuste según tu sistema)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
#os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"

mp_hands = mp.solutions.hands
hands1 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
# hands2 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detectar_gesto(contorno, area):
    AREA_UMBRAL = 15000
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

        if 0.8 <= solidity <= 0.95:
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            if 6 <= len(approx) <= 7 and 0.70 <= aspect_ratio <= 0.90:
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
        if 0.6 <= solidity <= 0.70:
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            if 8 <= len(approx) <= 9 and 1.0 <= aspect_ratio <= 1.46:
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

        if 0.67 <= solidity <= 0.75:
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h
            if 7 <= len(approx) <= 8 and 0.40 <= aspect_ratio <= 60:
                return True
        return False
    except:
        return False

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Avanzado de Detección de Gestos")
        self.root.configure(bg='#2C3E50')
        self.root.state('zoomed')  # Maximizar ventana

        # Estilo para los widgets
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#34495E')
        style.configure('Info.TLabel', background='#34495E', foreground='white', font=('Helvetica', 12))
        style.configure('Title.TLabel', background='#2C3E50', foreground='white', font=('Helvetica', 16, 'bold'))

        # Frame principal
        self.main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Frame para título
        self.title_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.title_frame.pack(fill='x', pady=10)
        
        self.title_label = ttk.Label(self.title_frame, 
                                   text="Sistema de Detección y Análisis de Gestos en Tiempo Real",
                                   style='Title.TLabel')
        self.title_label.pack()

        # Frame para las cámaras y visualización
        self.cameras_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.cameras_frame.pack(expand=True, fill='both')

        # Frame para cámara principal
        self.cam1_frame = ttk.Frame(self.cameras_frame, style='Custom.TFrame')
        self.cam1_frame.pack(side='left', expand=True, fill='both', padx=5)
        
        self.cam1_label = ttk.Label(self.cam1_frame, text="Cámara Principal", style='Info.TLabel')
        self.cam1_label.pack()
        
        self.label_cam1 = tk.Label(self.cam1_frame, bg='black')
        self.label_cam1.pack(expand=True, fill='both', padx=5, pady=5)

        # Frame para ROI
        self.roi_frame = ttk.Frame(self.cameras_frame, style='Custom.TFrame')
        self.roi_frame.pack(side='left', expand=True, fill='both', padx=5)
        
        self.roi_label = ttk.Label(self.roi_frame, text="Región de Interés", style='Info.TLabel')
        self.roi_label.pack()
        
        self.label_roi = tk.Label(self.roi_frame, bg='black')
        self.label_roi.pack(expand=True, fill='both', padx=5, pady=5)

        # Frame para plano XY
        self.xy_frame = ttk.Frame(self.cameras_frame, style='Custom.TFrame')
        self.xy_frame.pack(side='left', expand=True, fill='both', padx=5)
        
        self.xy_label = ttk.Label(self.xy_frame, text="Plano XY", style='Info.TLabel')
        self.xy_label.pack()
        
        self.label_xy = tk.Label(self.xy_frame, bg='black')
        self.label_xy.pack(expand=True, fill='both', padx=5, pady=5)

        # Frame para información
        self.info_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.info_frame.pack(fill='x', pady=10)

        # Labels para mostrar información
        self.gesture_label = ttk.Label(self.info_frame, text="Gesto: --", style='Info.TLabel')
        self.gesture_label.pack(side='left', padx=20)
        
        self.hand_state_label = ttk.Label(self.info_frame, text="Estado: --", style='Info.TLabel')
        self.hand_state_label.pack(side='left', padx=20)
        
        self.position_label = ttk.Label(self.info_frame, text="Posición: --", style='Info.TLabel')
        self.position_label.pack(side='left', padx=20)

        self.gestures_label = ttk.Label(self.info_frame, text="Paz: -- | Dino: -- | Dislike: --", style='Info.TLabel')
        self.gestures_label.pack(side='left', padx=20)
        
        self.time_label = ttk.Label(self.info_frame, text="Tiempo: --", style='Info.TLabel')
        self.time_label.pack(side='right', padx=20)

        # Botones de control
        self.control_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.control_frame.pack(fill='x', pady=10)
        
        self.btn_quit = ttk.Button(self.control_frame, text="Salir", command=self.close)
        self.btn_quit.pack(side='right', padx=5)

        # Inicializar cámaras
        self.cap1 = cv2.VideoCapture(0)
        if not self.cap1.isOpened():
            self.cap1 = cv2.VideoCapture(1)
            if not self.cap1.isOpened():
                print("No se pudo abrir la cámara principal")
                return

        self.roi_image = None
        self.update_frames()

    def create_xy_plane(self, pos_x, pos_y, size=400):
        plane = np.ones((size, size, 3), dtype=np.uint8) * 255
        center = size // 2
        
        # Dibujar ejes
        cv2.line(plane, (0, center), (size, center), (0, 0, 0), 2)  # Eje X
        cv2.line(plane, (center, 0), (center, size), (0, 0, 0), 2)  # Eje Y
        
        # Dibujar punto de la mano
        point_x = center + pos_x
        point_y = center + pos_y
        
        # Asegurar que el punto esté dentro de los límites
        point_x = max(0, min(size-1, point_x))
        point_y = max(0, min(size-1, point_y))
        
        cv2.circle(plane, (point_x, point_y), 8, (255, 0, 0), -1)
        return plane

    def update_frames(self):
        try:
            ret1, frame1 = self.cap1.read()
            if not ret1:
                return

            frame1 = cv2.flip(frame1, 1)
            rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = hands1.process(rgb_frame1)

            gesture_basic = "No detectado"
            pos_x = 0
            pos_y = 0
            paz_detected = False
            dino_detected = False
            dislike_detected = False

            output_frame1 = frame1.copy()
            roi = None

            if results1.multi_hand_landmarks:
                for hand_landmarks in results1.multi_hand_landmarks:
                    h, w, _ = frame1.shape
                    x_list = []
                    y_list = []
                    for lm in hand_landmarks.landmark:
                        x_list.append(int(lm.x * w))
                        y_list.append(int(lm.y * h))

                    x_min = max(0, min(x_list) - 40)
                    y_min = max(0, min(y_list) - 40)
                    x_max = min(w, max(x_list) + 40)
                    y_max = min(h, max(y_list) + 40)

                    # Calcular el centro del ROI
                    roi_center_x = (x_min + x_max) // 2
                    roi_center_y = (y_min + y_max) // 2
                    
                    # Calcular la posición relativa al centro de la imagen
                    pos_x = roi_center_x - (w // 2)
                    pos_y = roi_center_y - (h // 2)

                    roi = frame1[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue

                    # Procesar ROI
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
                            paz_detected = detectar_paz(contorno_mano)
                            dino_detected = detectar_dino(contorno_mano)
                            dislike_detected = detectar_dislike(contorno_mano)

                    cv2.rectangle(output_frame1, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.circle(output_frame1, (roi_center_x, roi_center_y), 5, (0, 255, 255), -1)

            # Crear y actualizar plano XY
            xy_plane = self.create_xy_plane(pos_x, pos_y)
            xy_img = Image.fromarray(xy_plane)
            xy_imgtk = ImageTk.PhotoImage(image=xy_img)
            self.label_xy.imgtk = xy_imgtk
            self.label_xy.configure(image=xy_imgtk)

            # Actualizar interfaz
            img1 = Image.fromarray(cv2.cvtColor(output_frame1, cv2.COLOR_BGR2RGB))
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.label_cam1.imgtk = imgtk1
            self.label_cam1.configure(image=imgtk1)

            if roi is not None:
                roi_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_imgtk = ImageTk.PhotoImage(image=roi_img)
                self.label_roi.imgtk = roi_imgtk
                self.label_roi.configure(image=roi_imgtk)

            # Actualizar etiquetas de información
            self.gesture_label.configure(text=f"Gesto: {gesture_basic}")
            self.hand_state_label.configure(text=f"Estado: {'Mano detectada' if results1.multi_hand_landmarks else 'Sin detección'}")
            self.position_label.configure(text=f"Posición: ({pos_x}, {pos_y})")
            self.gestures_label.configure(text=f"Paz: {paz_detected} | Dino: {dino_detected} | Dislike: {dislike_detected}")
            self.time_label.configure(text=f"Tiempo: {datetime.datetime.now().strftime('%H:%M:%S')}")

            self.root.after(10, self.update_frames)

        except Exception as e:
            print(f"Error en update_frames: {e}")
            print(traceback.format_exc())

    def close(self):
        if self.cap1.isOpened():
            self.cap1.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
