#!/usr/bin/env python3

#--------------------------------------------------------------------------------------------------
# IMPORTAMOS LAS LIBRERIAS NECESARIAS
#--------------------------------------------------------------------------------------------------
#importamos OpenCV. Además de todas las funcionalidades de vision por computadora
#OpenCV cuenta con el modelo de Haar Cascades para la detección de rostros
import cv2

#importamos sys que es el paquete que nos permite interactuar con el sistema operativo
import sys


#--------------------------------------------------------------------------------------------------
# COMIENZO DEL PROCESAMIENTO
#--------------------------------------------------------------------------------------------------
#Creamos un objeto detector que utiliza el algoritmo Haar Cascade para detección de objetos
#Argumento: haarcascade_frontalface_default.xml es el modelo pre-entrenado para rostros frontales

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Utilizamos la libreria OpenCV para leer imagen de un archivo
image = cv2.imread('faces.jpg')

#La detección por haar cascades solamente trabaja en escala de grises
#Utilizamos la conversión de OpenCV para hacer el trabajo:
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#--------------------------------------------
# AQUI SE HACE REALMENTE LA DETECCIÓN FACIAL:
#--------------------------------------------
# detectMultiScale() es el método principal para realizar detecciones con clasificadores Haar Cascade. 
# Su función es encontrar objetos en una imagen a diferentes escalas.
# Devuelve una lista de detecciones, puede ser una lista vacía si no hay detecciones o una lista de 
# con formato: [[x0,y0,w0,h0],...,[xn,yn,wn,hn]] para n rostros detectados donde
# xi es la coordenada x del rostro i en pxs
# yi es la coordenada y del rostro i en pxs
# wi y hi son ancho y alto en pxs del rostro (este clasificador considera que los rostros son cuadrados es decir wi = hi para todos los rostros)

faces = detector.detectMultiScale(
    image,           # Imagen de entrada (escala de grises)
    scaleFactor=1.1, # Factor de escala entre pirámides
    minNeighbors=5,  # Mínimo de vecinos para confirmar detección
    minSize=(30,30), # Tamaño mínimo del objeto
    maxSize=None,    # Tamaño máximo del objeto  
    flags=None       # Flags de comportamiento
)

#Imprimimos los resultados en pantalla
print("----------------------------------------------------------------------")
print(f"Se han detectado {len(faces)} rostros en la imagen proporcionada")
print("----------------------------------------------------------------------")

print("Coordenadas y dimensiones de los rostros en px:")

n=1
for (x, y, w, h) in faces:
    print(f"Imagen {n}: X:{x}, Y:{y}, W:{w}")
    n+=1
    
    
