from flask import Flask, Response
import cv2
import time
import numpy as np

#===================================================================================
# SERVIDOR WEB FLASK
#===================================================================================

# Inicializar Flask
app = Flask(__name__)

# Variables globales para comunicarse con el detector principal
obtener_datos = None
frame_lock = None


def generar_frames():
    """
    Generador de frames para el stream HTTP
    Toma los frames del detector principal y los convierte en stream JPEG
    """
    while True:
        # Obtener el frame más reciente del detector
        frame_actual, _ = obtener_datos()
        
        if frame_actual is not None:
            frame = frame_actual
        else:
            # Frame negro si no hay datos disponibles
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Codificar como JPEG para enviar por HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Formato multipart para streaming continuo
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)  # ~20 FPS para el stream web


@app.route('/')
def index():
    """
    Página principal con video stream embebido
    """
    return """
    <html>
    <head>
        <title>Detección de Rostros - Raspberry Pi</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #333; 
                text-align: center;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
            }
            .video-stream {
                border: 3px solid #333;
                border-radius: 10px;
                max-width: 100%;
            }
            .nav-links {
                text-align: center;
                margin-top: 20px;
            }
            .nav-links a {
                color: #007bff;
                text-decoration: none;
                margin: 0 10px;
                padding: 10px 20px;
                border: 1px solid #007bff;
                border-radius: 5px;
                display: inline-block;
            }
            .nav-links a:hover {
                background-color: #007bff;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Detección de Rostros en Tiempo Real</h1>
            <p style="text-align: center;">Sistema de visión por computadora con Raspberry Pi</p>
            
            <div class="video-container">
                <img src="/video_feed" class="video-stream" alt="Stream de video">
            </div>
            
            <div class="nav-links">
                <a href="/stats">📊 Ver estadísticas</a>
                <a href="/video_feed">🎬 Solo video</a>
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
                <h3>ℹ️ Información del sistema:</h3>
                <ul>
                    <li>Resolución: 640x480 píxeles</li>
                    <li>Zona de detección: 300 píxeles de ancho (centro)</li>
                    <li>Algoritmo: Haar Cascade (OpenCV)</li>
                    <li>Los rectángulos verdes indican rostros detectados</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    """
    Endpoint que devuelve el stream de video en formato MJPEG
    """
    return Response(generar_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """
    Página de estadísticas con información de detecciones
    """
    # Obtener datos actuales del detector
    _, detecciones_actuales = obtener_datos()
    
    num_rostros = len(detecciones_actuales)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Formatear coordenadas para mostrar
    coordenadas_str = ""
    if detecciones_actuales is not None and len(detecciones_actuales) > 0:
        for i, (x, y, w, h) in enumerate(detecciones_actuales):
            coordenadas_str += f"Rostro {i+1}: x={x}, y={y}, ancho={w}, alto={h}\n"
    else:
        coordenadas_str = "No se detectaron rostros"
    
    return f"""
    <html>
    <head>
        <title>Estadísticas - Detección de Rostros</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h2 {{ 
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            .stat-box {{
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
            .stat-label {{
                font-weight: bold;
                color: #495057;
            }}
            .stat-value {{
                font-size: 1.2em;
                color: #007bff;
                margin-left: 10px;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 3px;
                border: 1px solid #dee2e6;
                overflow-x: auto;
            }}
            .back-link {{
                text-align: center;
                margin-top: 30px;
            }}
            .back-link a {{
                color: #007bff;
                text-decoration: none;
                padding: 10px 20px;
                border: 1px solid #007bff;
                border-radius: 5px;
            }}
            .back-link a:hover {{
                background-color: #007bff;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📊 Estadísticas de Detección</h2>
            
            <div class="stat-box">
                <span class="stat-label">🕒 Última actualización:</span>
                <span class="stat-value">{timestamp}</span>
            </div>
            
            <div class="stat-box">
                <span class="stat-label">👤 Rostros detectados:</span>
                <span class="stat-value">{num_rostros}</span>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">📍 Coordenadas de detección:</div>
                <pre>{coordenadas_str}</pre>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">ℹ️ Información técnica:</div>
                <ul>
                    <li>Las coordenadas están en píxeles relativos a la zona de interés</li>
                    <li>x, y: esquina superior izquierda del rectángulo</li>
                    <li>ancho, alto: dimensiones del rectángulo detectado</li>
                </ul>
            </div>
            
            <div class="back-link">
                <a href="/">← Volver al video en vivo</a>
            </div>
        </div>
    </body>
    </html>
    """


def inicializar_servidor(func_obtener_datos, lock):
    """
    Función para inicializar el servidor Flask desde el programa principal
    
    Args:
        func_obtener_datos: Función que devuelve (frame, detecciones)
        lock: Lock para sincronización de threads
    """
    global obtener_datos, frame_lock
    
    obtener_datos = func_obtener_datos
    frame_lock = lock
    
    print("🌐 Servidor web disponible en: http://0.0.0.0:5000")
    print("📱 Desde otros dispositivos: http://[IP_DE_TU_RASPBERRY]:5000")
    print("🛑 Presiona Ctrl+C para detener el sistema")
    
    # Iniciar servidor Flask
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🔴 Servidor detenido por el usuario")
