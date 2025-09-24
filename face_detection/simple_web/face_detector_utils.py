from picamera2 import Picamera2
import cv2
import threading
import numpy as np

#===================================================================================
# CONFIGURACIÓN Y CONSTANTES
#===================================================================================
class Config:
    """Configuración centralizada del sistema"""
    # Resolución del video
    ANCHO_VIDEO = 640
    ALTO_VIDEO = 480
    
    # Zona de interés (región central donde buscar rostros)
    ANCHO_ZONA_INTERES = 150
    
    # Frames por segundo
    FPS = 5
    
    # Parámetros del detector Haar Cascade
    SCALE_FACTOR = 1.2      # Factor de escala para detección multi-escala
    MIN_NEIGHBORS = 4       # Mínimo vecinos para validar detección
    MIN_SIZE = (40, 40)     # Tamaño mínimo del rostro en píxeles
    
    # Archivo del modelo pre-entrenado
    HAAR_MODEL = "haarcascade_frontalface_alt2.xml"


#===================================================================================
# CLASE PRINCIPAL - DETECTOR DE ROSTROS
#===================================================================================
class FaceDetector:
    """
    Clase que encapsula toda la funcionalidad de detección de rostros
    """
    
    def __init__(self):
        """Inicializa la cámara, el detector y las variables globales"""
        
        # Variables para compartir datos entre threads
        self.ultimo_frame = None
        self.ultimas_detecciones = []
        self.frame_lock = threading.Lock()
        
        # Inicializar cámara Picamera2
        self._inicializar_camara()
        
        # Cargar modelo de detección de rostros
        self._cargar_detector()
        
    def _inicializar_camara(self):
        """Configura e inicia la cámara"""
        print(f"🔧 Configurando cámara: {Config.ANCHO_VIDEO}x{Config.ALTO_VIDEO} @ {Config.FPS}fps")
        
        self.picam2 = Picamera2()
        
        config = self.picam2.create_preview_configuration(
            main={
                "size": (Config.ANCHO_VIDEO, Config.ALTO_VIDEO),
                "format": "RGB888"  # Usar RGB888 que OpenCV interpreta como BGR
            },
            controls={"FrameRate": Config.FPS}
        )
        
        self.picam2.configure(config)
        self.picam2.start()
        print("✅ Cámara iniciada correctamente")
        
    def _cargar_detector(self):
        """Carga el clasificador Haar para detección de rostros"""
        print(f"🧠 Cargando modelo: {Config.HAAR_MODEL}")
        
        self.face_cascade = cv2.CascadeClassifier(Config.HAAR_MODEL)
        
        if self.face_cascade.empty():
            raise FileNotFoundError(f"❌ No se pudo cargar {Config.HAAR_MODEL}")
            
        print("✅ Modelo de detección cargado")
    
    def capturar_frame(self):
        """
        Captura un frame de la cámara
        Returns: numpy array con la imagen BGR
        """
        return self.picam2.capture_array()
    
    def preprocesar_frame(self, frame):
        """
        Preprocesa el frame para la detección:
        1. Recorta la zona de interés (centro de la imagen)
        2. Convierte a escala de grises
        
        Args:
            frame: Frame original de la cámara
        Returns:
            gray_crop: Zona recortada en escala de grises
        """
        # Calcular coordenadas para centrar la zona de interés
        x0 = (Config.ANCHO_VIDEO - Config.ANCHO_ZONA_INTERES) // 2
        y0 = 0
        
        # Recortar zona de interés
        frame_crop = frame[y0:y0+Config.ALTO_VIDEO, x0:x0+Config.ANCHO_ZONA_INTERES]
        
        # Convertir a escala de grises (requerido por Haar Cascade)
        gray_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        
        return gray_crop
    
    def detectar_rostros(self, gray_image):
        """
        Detecta rostros en la imagen en escala de grises usando Haar Cascade
        
        Args:
            gray_image: Imagen en escala de grises
        Returns:
            faces: Lista de tuplas (x, y, w, h) con las coordenadas de los rostros
        """
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=Config.SCALE_FACTOR,
            minNeighbors=Config.MIN_NEIGHBORS,
            minSize=Config.MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def dibujar_detecciones(self, frame_original, faces):
        """
        Dibuja rectángulos alrededor de los rostros detectados
        
        Args:
            frame_original: Frame completo de la cámara
            faces: Lista de rostros detectados [(x,y,w,h), ...]
        Returns:
            frame con rectángulos dibujados
        """
        # Calcular offset para ajustar coordenadas de la zona de interés
        x_offset = (Config.ANCHO_VIDEO - Config.ANCHO_ZONA_INTERES) // 2
        y_offset = 0
        
        # Hacer una copia del frame para no modificar el original
        frame_con_detecciones = frame_original.copy()
        
        # Dibujar rectángulo de la zona de interés (opcional, para visualización)
        cv2.rectangle(
            frame_con_detecciones,
            (x_offset, y_offset),
            (x_offset + Config.ANCHO_ZONA_INTERES, y_offset + Config.ALTO_VIDEO),
            (255, 255, 0),  # Amarillo para la zona de interés
            1
        )
        
        # Dibujar rectángulos alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            # Ajustar coordenadas al frame completo
            x_global = x + x_offset
            y_global = y + y_offset
            
            # Dibujar rectángulo verde alrededor del rostro
            cv2.rectangle(
                frame_con_detecciones,
                (x_global, y_global),
                (x_global + w, y_global + h),
                (0, 255, 0),  # Verde para rostros detectados
                2
            )
            
            # Opcional: agregar etiqueta con el número de rostro
            cv2.putText(
                frame_con_detecciones,
                f"Rostro",
                (x_global, y_global - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        return frame_con_detecciones
    
    def actualizar_datos_globales(self, frame, detecciones):
        """
        Actualiza las variables globales de manera thread-safe
        
        Args:
            frame: Frame procesado con detecciones dibujadas
            detecciones: Lista de rostros detectados
        """
        with self.frame_lock:
            self.ultimo_frame = frame
            self.ultimas_detecciones = detecciones
    
    def obtener_datos_actuales(self):
        """
        Obtiene los datos actuales de manera thread-safe
        
        Returns:
            tuple: (ultimo_frame, ultimas_detecciones)
        """
        with self.frame_lock:
            return self.ultimo_frame, self.ultimas_detecciones
    
    def cerrar_camara(self):
        """Cierra la cámara correctamente"""
        print("📷 Cerrando cámara...")
        self.picam2.stop()
        print("✅ Cámara cerrada")
    
    def obtener_estadisticas(self):
        """
        Obtiene estadísticas del sistema
        
        Returns:
            dict: Diccionario con estadísticas
        """
        with self.frame_lock:
            stats = {
                'num_rostros': len(self.ultimas_detecciones) if self.ultimas_detecciones is not None else 0,
                'detecciones': self.ultimas_detecciones,
                'resolucion': f"{Config.ANCHO_VIDEO}x{Config.ALTO_VIDEO}",
                'zona_interes': f"{Config.ANCHO_ZONA_INTERES}px",
                'fps': Config.FPS
            }
        return stats


#===================================================================================
# FUNCIONES AUXILIARES (si se necesitan fuera de la clase)
#===================================================================================

def validar_modelo_haar(ruta_modelo):
    """
    Valida que el archivo del modelo Haar existe y se puede cargar
    
    Args:
        ruta_modelo: Ruta al archivo .xml del modelo
    Returns:
        bool: True si el modelo es válido
    """
    import os
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: No se encuentra el archivo {ruta_modelo}")
        return False
    
    cascade = cv2.CascadeClassifier(ruta_modelo)
    if cascade.empty():
        print(f"❌ Error: No se pudo cargar el modelo {ruta_modelo}")
        return False
    
    print(f"✅ Modelo {ruta_modelo} validado correctamente")
    return True


def mostrar_info_sistema():
    """Muestra información del sistema de detección"""
    print("\n" + "="*50)
    print("📋 INFORMACIÓN DEL SISTEMA")
    print("="*50)
    print(f"📹 Resolución: {Config.ANCHO_VIDEO}x{Config.ALTO_VIDEO}")
    print(f"🎯 Zona de interés: {Config.ANCHO_ZONA_INTERES}px de ancho")
    print(f"⚡ FPS objetivo: {Config.FPS}")
    print(f"🧠 Modelo: {Config.HAAR_MODEL}")
    print(f"🔍 Tamaño mínimo rostro: {Config.MIN_SIZE}")
    print("="*50 + "\n")


# Ejecutar validación al importar el módulo
if __name__ == "__main__":
    mostrar_info_sistema()
    validar_modelo_haar(Config.HAAR_MODEL)
