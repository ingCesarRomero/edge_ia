#------------------------------------------------------------------
# FUNCIONES UTILITARIAS
#------------------------------------------------------------------
import cv2

def dibujar_rostros(imagen,rostros,nombre_archivo):
    image_with_boxes = imagen.copy()
    for i, (x, y, w, h) in enumerate(rostros):
        # Dibujar rectángulo verde alrededor del rostro
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Dibujar fondo para el texto
        cv2.rectangle(image_with_boxes, (x, y-25), (x+60, y), (0, 255, 0), -1)
        
        # Añadir etiqueta con número de rostro
        cv2.putText(image_with_boxes, f'# {i+1}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                   
    cv2.imwrite(nombre_archivo, image_with_boxes)
    
