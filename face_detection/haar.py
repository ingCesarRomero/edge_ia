import cv2
import time
# Parámetros
max_width = 600
max_height = 600

# Ruta directa al archivo
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread("cesar.jpg")
orig_h, orig_w = image.shape[:2]
# Redimensionar manteniendo relación de aspecto
scale = min(max_width / orig_w, max_height / orig_h, 1.0)  # nunca agrandar
new_w = int(orig_w * scale)
new_h = int(orig_h * scale)
image_resized = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_AREA)


start=time.time()
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
print(time.time()-start)
for (x,y,w,h) in faces:
    cv2.rectangle(image_resized, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imwrite("result_haar.jpg", image_resized)
print(f"{len(faces)} caras detectadas. Resultado guardado en result_haar.jpg")
