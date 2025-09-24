import threading
import time
from face_detector_utils import FaceDetector
from flask_server  import inicializar_servidor

#===================================================================================
# PROGRAMA PRINCIPAL - DETECCIÓN DE ROSTROS EN TIEMPO REAL
#===================================================================================

def deteccion_continua():
    global detector
    
    print("🔄 Iniciando detección continua de rostros...")
    frame_count = 0
    
    try:
        while True:
            # 1. CAPTURAR FRAME DE LA CÁMARA
            escena = detector.capturar_frame()
            
            # 2. PREPROCESAR: recortar zona de interés y convertir a gris
            zona_de_interes = detector.preprocesar_frame(escena)

            # 3. DETECTAR ROSTROS usando el algoritmo Haar Cascade y procesar
            rostros_detectados = detector.detectar_rostros(zona_de_interes)
            #===================================================================================#
            #           AQUI LA LOGICA DE LO QUE QUIERAS HACER CON LOS ROSTROS DETECTADOS       #
            #                                ***********                                        #
            #                                   *****                                           #
            #                                     *                                             #
            #===================================================================================#           

            # 4. DIBUJAR RECTÁNGULOS alrededor de los rostros
            escena_con_detecciones = detector.dibujar_detecciones(escena, rostros_detectados)



            #===================================================================================#
            #                                     *                                             #
            #                                   *****                                           #
            #                                ***********                                        #
            #                   FIN LOGICA DE PROCESAMIENTO DE ROSTROS                          #
            #===================================================================================#   


            # 5. ACTUALIZAR datos para el servidor web (thread-safe)
            detector.actualizar_datos_globales(escena_con_detecciones, rostros_detectados)
            
            # Contador y pequeña pausa
            frame_count += 1
            if frame_count % 100 == 0:  # Mostrar progreso cada 100 frames
                print(f"📊 Frames procesados: {frame_count}, Rostros: {len(rostros_detectados)}")
            
            time.sleep(0.01)  # Pausa para no saturar el procesador
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo detección...")
        detector.cerrar_camara()


def obtener_datos_camara():
    """
    Función para que el servidor web obtenga los datos actualizados
    """
    return detector.obtener_datos_actuales()


if __name__ == '__main__':
    print("=" * 60)
    print("🎥 SISTEMA DE DETECCIÓN DE ROSTROS - RASPBERRY PI")
    print("=" * 60)
    
    # PASO 1: Inicializar el detector de rostros
    print("📷 Inicializando detector de rostros...")
    detector = FaceDetector()
    print("✅ Detector inicializado correctamente")
    
    # PASO 2: Iniciar la detección en un hilo separado
    print("🚀 Iniciando thread de detección...")
    detector_thread = threading.Thread(target=deteccion_continua)
    detector_thread.daemon = True
    detector_thread.start()
    print("✅ Detección iniciada en segundo plano")
    
    # PASO 3: Iniciar servidor web para visualización
    print("🌐 Iniciando interfaz web...")
    inicializar_servidor(obtener_datos_camara, detector.frame_lock)
