from picamera2 import Picamera2
import cv2
import time

# Inicializar cámara
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1024,768)})
picam2.configure(config)
picam2.start()

# Cargar Haar
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frame_count = 0

try:
    while True:
        frame = picam2.capture_array()  # devuelve frame como numpy array BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
        )

        for (x, y, w, h) in faces:
            cx = x + w // 2
            cy = y + h // 2
            print(f"Frame {frame_count}: Face center at ({cx},{cy})")

        frame_count += 1
        time.sleep(0.05)  # opcional: limitar FPS

except KeyboardInterrupt:
    print("Cerrando cámara...")
    picam2.stop()