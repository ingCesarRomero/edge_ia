#!/bin/bash

# test_tflite_yolo.sh - Test automático de TensorFlow Lite con YOLOv8n
# Script para verificar que la instalación de IA en el Edge funciona correctamente

# Configuración
TEST_DIR="$HOME/test_yolo"
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.tflite"
IMAGE_URL="https://ultralytics.com/images/bus.jpg"
SCRIPT_URL="https://gist.githubusercontent.com/tu-usuario/raw/main/test_deteccion.py"

echo "=================================================="
echo "    TEST AUTOMÁTICO TENSORFLOW LITE + YOLO"
echo "    Para Raspberry Pi Zero 2 W"
echo "=================================================="
echo ""

# Crear directorio de prueba
echo "📁 Creando directorio de prueba en: $TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Descargar archivos necesarios
echo "📦 Descargando archivos necesarios..."

echo "   Descargando modelo YOLOv8n..."
wget -q --show-progress "$MODEL_URL" -O yolov8n.tflite

echo "   Descargando imagen de prueba..."
wget -q --show-progress "$IMAGE_URL" -O bus.jpg

# Crear script de prueba Python
echo "   Creando script de detección..."
cat > test_deteccion.py << 'EOF'
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuración
model_path = 'yolov8n.tflite'
image_path = 'bus.jpg'
conf_threshold = 0.5

# Nombres de las clases COCO (80 objetos)
class_names = [
    'persona', 'bicicleta', 'auto', 'motocicleta', 'avión', 'autobús', 'tren', 'camión', 'bote',
    'semáforo', 'hidrante', 'señal de stop', 'parquímetro', 'banco', 'pájaro', 'gato',
    'perro', 'caballo', 'oveja', 'vaca', 'elefante', 'oso', 'cebra', 'jirafa', 'mochila',
    'paraguas', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquís', 'snowboard', 'pelota deportiva',
    'cometa', 'bate de béisbol', 'guante de béisbol', 'skateboard', 'tabla de surf', 'raqueta de tenis',
    'botella', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara', 'tazón', 'banana', 'manzana',
    'sándwich', 'naranja', 'brócoli', 'zanahoria', 'hot dog', 'pizza', 'dona', 'pastel',
    'silla', 'sofá', 'planta en maceta', 'cama', 'mesa de comedor', 'inodoro', 'televisor', 'computadora portátil',
    'ratón', 'control remoto', 'teclado', 'teléfono móvil', 'microondas', 'horno', 'tostadora', 'fregadero',
    'refrigerador', 'libro', 'reloj', 'florero', 'tijeras', 'oso de peluche', 'secador de pelo', 'cepillo de dientes'
]

print("🔍 Iniciando detección de objetos con YOLOv8n + TensorFlow Lite")
print("📦 Modelo: yolov8n.tflite")
print("🖼️ Imagen: bus.jpg")
print("---")

try:
    # 1. Cargar el modelo TFLite
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 2. Obtener detalles de entrada/salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    # 3. Cargar y preprocesar la imagen
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((width, height))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # 4. Ejecutar la inferencia
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    # 5. Obtener resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)
    
    # 6. Procesar detecciones
    detections_found = []
    
    for detection in output_data:
        x_center, y_center, w, h, confidence = detection[:5]
        
        if confidence > conf_threshold:
            class_probs = detection[5:85]
            class_id = np.argmax(class_probs)
            class_name = class_names[class_id]
            
            detections_found.append({
                'objeto': class_name,
                'confianza': float(confidence),
                'clase_id': int(class_id)
            })
    
    # 7. Mostrar resultados
    if detections_found:
        print("✅ Objetos detectados:")
        print("=====================")
        
        detections_found.sort(key=lambda x: x['confianza'], reverse=True)
        
        for i, det in enumerate(detections_found, 1):
            print(f"{i}. {det['objeto'].upper()} - {det['confianza']:.1%} de confianza")
        
        print("")
        print(f"📊 Total: {len(detections_found)} objetos detectados")
        
    else:
        print("❌ No se detectaron objetos con confianza suficiente")
        
except Exception as e:
    print(f"❌ Error durante la detección: {str(e)}")
    print("Asegúrate de que:")
    print("1. El archivo yolov8n.tflite existe en el directorio")
    print("2. El archivo bus.jpg existe en el directorio")
    print("3. TensorFlow Lite está instalado correctamente")
EOF

echo "✅ Todos los archivos descargados y creados"
echo ""

# Verificar que los archivos existen
if [ ! -f "yolov8n.tflite" ] || [ ! -f "bus.jpg" ]; then
    echo "❌ Error: No se pudieron descargar todos los archivos"
    exit 1
fi

# Ejecutar la prueba
echo "🚀 Ejecutando prueba de detección..."
echo ""

# Activar el entorno virtual si existe
if [ -f "$HOME/mi_proyecto_ia/venv/bin/activate" ]; then
    source "$HOME/mi_proyecto_ia/venv/bin/activate"
    echo "🐍 Entorno virtual activado"
fi

python test_deteccion.py

echo ""
echo "=================================================="
echo "    TEST COMPLETADO"
echo "📍 Directorio de prueba: $TEST_DIR"
echo "📝 Para ejecutar nuevamente:"
echo "    cd $TEST_DIR && python test_deteccion.py"
echo "=================================================="