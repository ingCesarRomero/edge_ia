import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

coco_classes = \
{
	0:   "person",
	1:   "bicycle",
	2:   "car",
	3:   "motorcycle",
	4:   "airplane",
	5:   "bus",
	6:   "train",
	7:   "truck",
	8:   "boat",
	9:   "traffic light",
	10:  "fire hydrant",
	12:  "stop sign",
	13:  "parking meter",
	14:  "bench",
	15:  "bird",
	16:  "cat",
	17:  "dog",
	18:  "horse",
	19:  "sheep",
	20:  "cow",
	21:  "elephant",
	22:  "bear",
	23:  "zebra",
	24:  "giraffe",
	26:  "backpack",
	27:  "umbrella",
	30:  "handbag",
	31:  "tie",
	32:  "suitcase",
	33:  "frisbee",
	34:  "skis",
	35:  "snowboard",
	36:  "sports ball",
	37:  "kite",
	38:  "baseball bat",
	39:  "baseball glove",
	40:  "skateboard",
	41:  "surfboard",
	42:  "tennis racket",
	43:  "bottle",
	45:  "wine glass",
	46:  "cup",
	47:  "fork",
	48:  "knife",
	49:  "spoon",
	50:  "bowl",
	51:  "banana",
	52:  "apple",
	53:  "sandwich",
	54:  "orange",
	55:  "broccoli",
	56:  "carrot",
	57:  "hot dog",
	58:  "pizza",
	59:  "donut",
	60:  "cake",
	61:  "chair",
	62:  "couch",
	63:  "potted plant",
	64:  "bed",
	66:  "dining table",
	69:  "toilet",
	71:  "tv",
	72:  "laptop",
	73:  "mouse",
	74:  "remote",
	75:  "keyboard",
	76:  "cell phone",
	77:  "microwave",
	78:  "oven",
	79:  "toaster",
	80:  "sink",
	81:  "refrigerator",
	83:  "book",
	84:  "clock",
	85:  "vase",
	86:  "scissors",
	87:  "teddy bear",
	88:  "hair drier",
	89:  "toothbrush",
}
def process_single_image(image_path, output_path, conf_threshold=0.5):
    """Procesa una sola imagen con bounding boxes"""
    
    # Cargar modelo
    interpreter = Interpreter(model_path='mobilenet_v2_compatible.tflite')
    interpreter.allocate_tensors()
    
    # Obtener detalles del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    model_height, model_width = input_shape[1], input_shape[2]
    
    # Cargar imagen original
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    # Preprocesar
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)
    
    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Obtener resultados
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # Dibujar bounding boxes
    for i in range(len(scores)):
        if scores[i] > conf_threshold:
            
            class_id = int(classes[i])
            score = float(scores[i])
            print(f"Clase_id:{class_id}, clase:{coco_classes[class_id]}, confianza:{score}")
            # Convertir coordenadas
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * original_width)
            xmax = int(xmax * original_width)
            ymin = int(ymin * original_height)
            ymax = int(ymax * original_height)
            
            # Elegir color según clase
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Verde para personas, Azul para otros
            
            # Dibujar rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Dibujar etiqueta
            label = f"Class {class_id}: {score:.1%}"
            cv2.putText(image, label, (xmin, ymin-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Guardar imagen
    cv2.imwrite(output_path, image)
    print(f"✅ Imagen guardada: {output_path}")

# Ejemplo de uso
process_single_image('test.jpg', 'detecciones.jpg')
