import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import time
import cv2

# Configuración
MODEL_PATH = 'ssd_mobilenet_v2_coco_quant_postprocess.tflite'
LABELS_PATH = 'coco_labels.txt'
NUM_THREADS = 4  # Usar los 4 núcleos de la Pi Zero 2W

def load_labels(labels_path):
    """Cargar las etiquetas desde el archivo"""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def initialize_interpreter(model_path, num_threads):
    """Inicializar el intérprete de TensorFlow Lite"""
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

def get_model_details(interpreter):
    """Obtener información de entrada/salida del modelo"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Dimensiones que espera el modelo
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    return input_details, output_details, width, height

def preprocess_image(image_path, width, height):
    """Preparar la imagen para el modelo"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # Guardar tamaño original
    
    # Redimensionar a lo que espera el modelo
    image = image.resize((width, height))
    input_data = np.expand_dims(image, axis=0)
    
    # Convertir al tipo correcto (uint8 para modelos cuantizados)
    input_data = np.uint8(input_data)
    
    return input_data, original_size

def detect_objects(interpreter, input_details, output_details, input_data):
    """Realizar la detección de objetos"""
    # Medir tiempo de inferencia
    start_time = time.monotonic()
    
    # Pasar la imagen al modelo
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    inference_time = (time.monotonic() - start_time) * 1000
    
    # Obtener resultados
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    return boxes, classes, scores, inference_time

def filter_detections(boxes, classes, scores, labels, confidence_threshold=0.25):
    """Filtrar detecciones por confianza"""
    detections = []
    
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            detection = {
                'class_id': int(classes[i]),
                'label': labels[int(classes[i])],
                'confidence': float(scores[i]),
                'box': boxes[i]  # [ymin, xmin, ymax, xmax]
            }
            detections.append(detection)
    
    return detections

def draw_detections(image_path, detections, output_path='resultado.jpg'):
    """Dibujar las detecciones en la imagen"""
    # Cargar imagen original
    image = Image.open(image_path)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Dibujar cada detección
    for detection in detections:
        label = f"{detection['label']}: {detection['confidence']:.2f}"
        ymin, xmin, ymax, xmax = detection['box']
        
        # Convertir coordenadas normalizadas a píxeles
        h, w, _ = img_cv.shape
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        
        # Dibujar rectángulo y texto
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Guardar resultado
    cv2.imwrite(output_path, img_cv)
    return output_path

def main():
    """Función principal"""
    print("🚀 Iniciando detector de objetos...")
    
    # Paso 1: Cargar etiquetas
    print("📋 Cargando etiquetas...")
    labels = load_labels(LABELS_PATH)
    print(f"   {len(labels)} etiquetas cargadas")
    
    # Paso 2: Inicializar modelo
    print("🧠 Cargando modelo TensorFlow Lite...")
    interpreter = initialize_interpreter(MODEL_PATH, NUM_THREADS)
    input_details, output_details, width, height = get_model_details(interpreter)
    print(f"   Modelo: {MODEL_PATH}")
    print(f"   Resolución: {width}x{height}")
    print(f"   Threads: {NUM_THREADS}")
    
    # Paso 3: Descargar imagen de prueba
    #print("📸 Descargando imagen de prueba...")
    #import urllib.request
    #test_image_url = "https://github.com/google-coral/test_data/raw/master/grace_hopper.bmp"
    #test_image_path = "test_image.bmp"
    #urllib.request.urlretrieve(test_image_url, test_image_path)
    #print(f"   Imagen guardada como: {test_image_path}")
    test_image_path = "personacaida2.webp"
    # Paso 4: Preprocesar imagen
    print("🖼️ Preprocesando imagen...")
    input_data, original_size = preprocess_image(test_image_path, width, height)
    print(f"   Tamaño original: {original_size}")
    print(f"   Tamaño modelo: {width}x{height}")
    
    # Paso 5: Realizar detección
    print("🔍 Detectando objetos...")
    boxes, classes, scores, inference_time = detect_objects(
        interpreter, input_details, output_details, input_data
    )
    
    # Paso 6: Filtrar resultados
    detections = filter_detections(boxes, classes, scores, labels)
    
    # Paso 7: Mostrar resultados
    print(f"\n✅ Resultados:")
    print(f"   Tiempo de inferencia: {inference_time:.2f} ms")
    print(f"   Detecciones encontradas: {len(detections)}")
    
    for i, detection in enumerate(detections):
        print(f"   {i+1}. {detection['label']}: {detection['confidence']:.2%}")
    
    # Paso 8: Dibujar y guardar resultados
    if detections:
        output_path = draw_detections(test_image_path, detections)
        print(f"\n💾 Imagen con detecciones guardada como: {output_path}")
    else:
        print("\n❌ No se encontraron objetos con confianza suficiente")
    
    print("\n🎉 Proceso completado!")

# Ejecutar el programa
if __name__ == "__main__":

    main()
