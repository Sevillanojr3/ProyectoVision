import cv2
import numpy as np
import mediapipe as mp
import os
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Deshabilitar transformaciones de hardware para evitar errores (ajuste según tu sistema)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"

mp_hands = mp.solutions.hands
hands1 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
hands2 = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

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

class MenuInicial:
    def __init__(self, root):
        self.root = root
        self.root.title("Selección de Cámaras")
        self.root.geometry("300x200")
        self.root.configure(bg='#2C3E50')
        
        ttk.Label(root, text="Seleccione el número de cámaras", 
                 style='Title.TLabel').pack(pady=20)
        
        ttk.Button(root, text="1 Cámara", 
                  command=lambda: self.iniciar_app(1)).pack(pady=10)
        ttk.Button(root, text="2 Cámaras", 
                  command=lambda: self.iniciar_app(2)).pack(pady=10)

    def iniciar_app(self, num_camaras):
        self.root.destroy()
        root_app = tk.Tk()
        App(root_app, num_camaras)
        root_app.mainloop()

class App:
    def __init__(self, root, num_camaras=2):
        self.root = root
        self.num_camaras = num_camaras
        self.root.title("Sistema Avanzado de Detección de Gestos")
        self.root.configure(bg='#2C3E50')
        self.root.state('zoomed')

        # Estilo para los widgets
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#34495E')
        style.configure('Info.TLabel', background='#34495E', foreground='white', font=('Helvetica', 10))
        style.configure('Title.TLabel', background='#2C3E50', foreground='white', font=('Helvetica', 14, 'bold'))

        # Frame principal
        self.main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # Frame para título
        self.title_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.title_frame.pack(fill='x', pady=5)
        
        self.title_label = ttk.Label(self.title_frame, 
                                   text="Sistema de Detección y Análisis de Gestos en Tiempo Real",
                                   style='Title.TLabel')
        self.title_label.pack()

        # Frame para las cámaras y visualización
        self.cameras_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.cameras_frame.pack(expand=True, fill='both')

        # Frame izquierdo para cámaras (60% del ancho)
        self.left_frame = ttk.Frame(self.cameras_frame, style='Custom.TFrame')
        self.left_frame.pack(side='left', expand=True, fill='both', padx=5)
        
        # Cámara principal (arriba)
        self.cam1_label = ttk.Label(self.left_frame, text="Cámara Principal", style='Info.TLabel')
        self.cam1_label.pack()
        self.label_cam1 = tk.Label(self.left_frame, bg='black')
        self.label_cam1.pack(padx=5, pady=5)

        # Cámara secundaria (abajo)
        self.cam2_label = ttk.Label(self.left_frame, text="Cámara Secundaria", style='Info.TLabel')
        self.cam2_label.pack()
        self.label_cam2 = tk.Label(self.left_frame, bg='black')
        self.label_cam2.pack(padx=5, pady=5)

        # Frame derecho para ROIs y gráfica (40% del ancho)
        self.right_frame = ttk.Frame(self.cameras_frame, style='Custom.TFrame', width=400, height=200)
        self.right_frame.pack(side='right', fill='both', padx=5)
        
        # Frame para ROIs (arriba)
        self.rois_frame = ttk.Frame(self.right_frame, style='Custom.TFrame', width=400, height=200)
        self.rois_frame.pack(fill='x', pady=5)
        
        # ROIs en horizontal
        self.roi1_frame = ttk.Frame(self.rois_frame, style='Custom.TFrame', width=200, height=200)
        self.roi1_frame.pack(side='left', padx=5)
        self.roi1_label = ttk.Label(self.roi1_frame, text="ROI 1", style='Info.TLabel')
        self.roi1_label.pack()
        self.label_roi1 = tk.Label(self.roi1_frame, bg='black')
        self.label_roi1.configure(width=150, height=150)
        self.label_roi1.pack()
        
        self.roi2_frame = ttk.Frame(self.rois_frame, style='Custom.TFrame', width=200, height=200)
        self.roi2_frame.pack(side='left', padx=5)
        self.roi2_label = ttk.Label(self.roi2_frame, text="ROI 2", style='Info.TLabel')
        self.roi2_label.pack()
        self.label_roi2 = tk.Label(self.roi2_frame, bg='black')
        self.label_roi2.configure(width=150, height=150)
        self.label_roi2.pack()

        # Frame para gráfica (abajo)
        self.plot_frame = ttk.Frame(self.right_frame, style='Custom.TFrame')
        self.plot_frame.pack(expand=True, fill='both', pady=5)
        
        self.fig = plt.Figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        # Frame para información
        self.info_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.info_frame.pack(fill='x', pady=5)

        # Labels para mostrar información
        self.gesture_label = ttk.Label(self.info_frame, text="Gesto: --", style='Info.TLabel')
        self.gesture_label.pack(side='left', padx=10)
        
        self.hand_state_label = ttk.Label(self.info_frame, text="Estado: --", style='Info.TLabel')
        self.hand_state_label.pack(side='left', padx=10)
        
        self.position_label = ttk.Label(self.info_frame, text="Posición: --", style='Info.TLabel')
        self.position_label.pack(side='left', padx=10)

        self.gestures_label = ttk.Label(self.info_frame, text="Paz: -- | Dino: -- | Dislike: --", style='Info.TLabel')
        self.gestures_label.pack(side='left', padx=10)
        
        self.time_label = ttk.Label(self.info_frame, text="Tiempo: --", style='Info.TLabel')
        self.time_label.pack(side='right', padx=10)

        # Botones de control
        self.control_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.control_frame.pack(fill='x', pady=5)
        
        self.btn_quit = ttk.Button(self.control_frame, text="Salir", command=self.close)
        self.btn_quit.pack(side='right', padx=5)

        # Inicializar cámaras
        self.cap1 = cv2.VideoCapture(0)
        if not self.cap1.isOpened():
            self.cap1 = cv2.VideoCapture(1)
            if not self.cap1.isOpened():
                print("No se pudo abrir la cámara principal")
                return

        self.cap2 = None
        if self.num_camaras == 2:
            self.cap2 = cv2.VideoCapture(1)
            if not self.cap2.isOpened():
                self.cap2 = cv2.VideoCapture(2)
                if not self.cap2.isOpened():
                    print("No se pudo abrir la cámara secundaria")
                    return

        self.roi_image1 = None
        self.roi_image2 = None
        self.update_frames()

    def update_3d_plot(self, x, y, z):
        self.ax.cla()
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([-300, 300])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.scatter([x], [y], [-z], c='red', marker='o')  # Invertimos el valor de z
        self.canvas.draw()

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
            pos_z = 0
            paz_detected = False
            dino_detected = False
            dislike_detected = False

            output_frame1 = frame1.copy()
            roi1 = None

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

                    roi_center_x = (x_min + x_max) // 2
                    roi_center_y = (y_min + y_max) // 2
                    
                    pos_x = roi_center_x - (w // 2)
                    pos_y = roi_center_y - (h // 2)

                    roi1 = frame1[y_min:y_max, x_min:x_max]
                    if roi1.size > 0:
                        gray_roi = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresh = cv2.bitwise_not(thresh)

                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            contorno_mano = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(contorno_mano)

                            if area > 3000:
                                cv2.drawContours(roi1, [contorno_mano], -1, (0, 255, 0), 3)
                                gesture_basic = detectar_gesto(contorno_mano, area)
                                paz_detected = detectar_paz(contorno_mano)
                                dino_detected = detectar_dino(contorno_mano)
                                dislike_detected = detectar_dislike(contorno_mano)

                    cv2.rectangle(output_frame1, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.circle(output_frame1, (roi_center_x, roi_center_y), 5, (0, 255, 255), -1)

            if self.num_camaras == 2 and self.cap2 is not None:
                ret2, frame2 = self.cap2.read()
                if ret2:
                    frame2 = cv2.flip(frame2, 1)
                    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    results2 = hands2.process(rgb_frame2)
                    output_frame2 = frame2.copy()
                    roi2 = None

                    if results2.multi_hand_landmarks:
                        for hand_landmarks in results2.multi_hand_landmarks:
                            h, w, _ = frame2.shape
                            x_list = []
                            y_list = []
                            for lm in hand_landmarks.landmark:
                                x_list.append(int(lm.x * w))
                                y_list.append(int(lm.y * h))

                            x_min = max(0, min(x_list) - 40)
                            y_min = max(0, min(y_list) - 40)
                            x_max = min(w, max(x_list) + 40)
                            y_max = min(h, max(y_list) + 40)

                            roi_center_y = (y_min + y_max) // 2
                            pos_z = roi_center_y - (h // 2)

                            roi2 = frame2[y_min:y_max, x_min:x_max]
                            cv2.rectangle(output_frame2, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                    img2 = Image.fromarray(cv2.cvtColor(output_frame2, cv2.COLOR_BGR2RGB))
                    img2 = img2.resize((320, 240), Image.LANCZOS)
                    imgtk2 = ImageTk.PhotoImage(image=img2)
                    self.label_cam2.imgtk = imgtk2
                    self.label_cam2.configure(image=imgtk2)

                    if roi2 is not None:
                        roi_img2 = Image.fromarray(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
                        roi_img2 = roi_img2.resize((150, 150), Image.LANCZOS)
                        roi_imgtk2 = ImageTk.PhotoImage(image=roi_img2)
                        self.label_roi2.configure(image=roi_imgtk2, width=150, height=150)
                        self.label_roi2.imgtk = roi_imgtk2
                    else:
                        blank_image = Image.new('RGB', (150, 150), color='black')
                        blank_imgtk = ImageTk.PhotoImage(image=blank_image)
                        self.label_roi2.configure(image=blank_imgtk, width=150, height=150)
                        self.label_roi2.imgtk = blank_imgtk

            # Actualizar gráfico 3D
            if self.num_camaras == 1:
                pos_z = 0
            self.update_3d_plot(pos_x, pos_y, pos_z)

            # Actualizar interfaz
            img1 = Image.fromarray(cv2.cvtColor(output_frame1, cv2.COLOR_BGR2RGB))
            img1 = img1.resize((320, 240), Image.LANCZOS)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.label_cam1.imgtk = imgtk1
            self.label_cam1.configure(image=imgtk1)

            if roi1 is not None:
                roi_img1 = Image.fromarray(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB))
                roi_img1 = roi_img1.resize((150, 150), Image.LANCZOS)
                roi_imgtk1 = ImageTk.PhotoImage(image=roi_img1)
                self.label_roi1.configure(image=roi_imgtk1, width=150, height=150)
                self.label_roi1.imgtk = roi_imgtk1
            else:
                blank_image = Image.new('RGB', (150, 150), color='black')
                blank_imgtk = ImageTk.PhotoImage(image=blank_image)
                self.label_roi1.configure(image=blank_imgtk, width=150, height=150)
                self.label_roi1.imgtk = blank_imgtk

            # Actualizar etiquetas de información
            self.gesture_label.configure(text=f"Gesto: {gesture_basic}")
            self.hand_state_label.configure(text=f"Estado: {'Mano detectada' if results1.multi_hand_landmarks else 'Sin detección'}")
            self.position_label.configure(text=f"Posición: ({pos_x}, {pos_y}, {pos_z})")
            self.gestures_label.configure(text=f"Paz: {paz_detected} | Dino: {dino_detected} | Dislike: {dislike_detected}")
            self.time_label.configure(text=f"Tiempo: {datetime.datetime.now().strftime('%H:%M:%S')}")

            self.root.after(10, self.update_frames)

        except Exception as e:
            print(f"Error en update_frames: {e}")
            print(traceback.format_exc())

    def close(self):
        if self.cap1.isOpened():
            self.cap1.release()
        if self.cap2 and self.cap2.isOpened():
            self.cap2.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    menu = MenuInicial(root)
    root.mainloop()
