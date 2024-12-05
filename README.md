# Control de un Brazo Robótico en Tiempo Real Mediante Visión por Computador Basada en Reconocimiento de Imágenes

## Integrantes:
- Gabriel Pulgar  
- Jesús Sevilla  
- Mishelle Quinchiguango  

---

## Objetivo:
Desarrollar un sistema integrado que permita el control de un brazo robótico en tiempo real utilizando visión por computador. Este sistema utilizará una cámara para capturar imágenes, procesarlas mediante algoritmos de reconocimiento y enviar comandos precisos al brazo robótico. 

## Descripción General:
Este proyecto busca facilitar la manipulación de brazos robóticos replicando los movimientos de un brazo humano, permitiendo así mayor flexibilidad para realizar diversas tareas de forma interactiva. Las operaciones se realizarán en tiempo real y de manera remota con el uso de cámaras. Las aplicaciones potenciales incluyen la manipulación de materiales peligrosos, operación de máquinas industriales, procedimientos médicos, entre otros.

---

## Etapas del Proyecto:

### **Etapa 1: Simulación de la Pinza con la Mano**
**Descripción:**  
Usar visión por computador para reconocer el gesto de la mano y simular los movimientos de apertura y cierre de la pinza del brazo robótico.

**Desafíos:**
- Reconocer correctamente el gesto de la mano en diferentes condiciones de luz.
- Mapear de forma precisa el gesto con el movimiento de la pinza del robot.

**Tareas:**
1. Captura de imágenes de la mano utilizando `cv2.VideoCapture()`.
2. Detección del gesto mediante filtrado de imagen y extracción de características.
3. Uso de `cv2.findContours()` para identificar la forma de la mano.
4. Análisis de huellas convexas con `cv2.convexHull()` para determinar si la mano está abierta o cerrada.
5. Traducción del resultado del reconocimiento en comandos para el movimiento de la pinza del brazo robótico.

---

### **Etapa 2: Movimiento en π Radianes**
**Descripción:**  
Controlar un movimiento básico del brazo robótico (rotación de π radianes).

**Desafíos:**
- Convertir la señal de reconocimiento visual en un ángulo de rotación preciso.
- Garantizar que el brazo ejecute el movimiento con precisión.

**Tareas:**
1. Detección del gesto de rotación:
   - Convertir la imagen a escala de grises usando `cv2.cvtColor()`.
   - Aplicar un umbral con `cv2.threshold()` para binarizar la imagen.
   - Encontrar los contornos con `cv2.findContours()` y calcular el ángulo de rotación usando `cv2.minAreaRect()`.
2. Conversión del ángulo de grados a radianes utilizando la fórmula matemática y la librería `math`.
3. Enviar los comandos de rotación al brazo robótico utilizando conocimientos de cinemática y robótica.

---

### **Etapa 3: Recoger y Mover un Objeto a una Posición Específica**
**Descripción:**  
Detectar un objeto, calcular su posición en el espacio y dirigir el brazo para recogerlo y moverlo a otra ubicación.

**Desafíos:**
- Detección y localización precisa del objeto.
- Control de la cinemática inversa del brazo robótico para evitar colisiones.
- Asegurar un agarre robusto para que el objeto no se caiga.

**Tareas:**
1. **Detección del objeto:**
   - Uso de algoritmos de detección de objetos, como `cv2.threshold()` o `cv2.Canny()`.
   - Localización del objeto mediante análisis de bordes y contornos.
2. **Detección de ángulos del brazo:**
   - Emplear `mediapipe` para detectar puntos clave y calcular los ángulos necesarios para el movimiento.
3. **Control del brazo:**
   - Implementar cinemática inversa para mover el brazo y agarrar el objeto con precisión.

---

## Tecnologías y Herramientas Utilizadas:
- **Librerías de Python:**
  - OpenCV: Procesamiento de imágenes, detección de contornos, análisis de gestos.
  - Mediapipe: Detección de puntos clave en la mano para calcular ángulos.
  - Math: Conversión de ángulos y cálculos trigonométricos.
- **Hardware:**
  - Brazo robótico programable.
  - Cámara para captura de video en tiempo real.
- **Otros conocimientos:**
  - Cinemática inversa.
  - Control de motores y actuadores robóticos.

---

## Cómo Ejecutar el Proyecto:
1. **Instalar dependencias:**
   - Asegúrate de tener Python instalado junto con las siguientes librerías:
     ```bash
     pip install opencv-python mediapipe
     ```
2. **Conectar el hardware:**
   - Configura la cámara para capturar video en tiempo real.
   - Asegúrate de que el brazo robótico esté conectado y configurado según las especificaciones del fabricante.
3. **Ejecutar cada etapa:**
   - Comienza ejecutando el script para la etapa 1. 
   - Avanza a las etapas 2 y 3, verificando la salida de cada una antes de proceder.

---

## Potenciales Aplicaciones:
- Manejo de materiales peligrosos.
- Automatización industrial.
- Asistencia en procesos médicos y quirúrgicos.
- Proyectos educativos y de investigación en robótica.

---

Con este proyecto, buscamos demostrar la integración efectiva entre visión por computador y robótica, abriendo posibilidades para una interacción humano-robot más intuitiva y adaptable.
