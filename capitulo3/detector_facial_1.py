#!/usr/bin/env python3

import cv2
import sys

#Creamos el detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Leemos imagen de un archivo
image = cv2.imread('faces.jpg')


#Convertimos la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Detectamos rostros en la imagen
faces = face_cascade.detectMultiScale(gray, 1.1, 5)


#Imprimimos los resultados en pantalla
n=1

for (x, y, w, h) in faces:
    print(f"Imagen {n}: X:{x}, Y:{y}, W:{w}")
    n+=1
    