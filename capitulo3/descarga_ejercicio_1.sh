#!/bin/bash

# descargar_capitulo3.sh - Descarga archivos específicos del capítulo 3

echo "📥 Descargando archivos del capítulo 3..."

# Lista de archivos conocidos (puedes agregar más)
ARCHIVOS=(
    "faces.jpg"
    "detector_facial_1.py"
    "haarcascade_frontalface_alt2.xml"
    "haarcascade_frontalface_default.xml"
    "haarcascade_profileface.xml"

)

for archivo in "${ARCHIVOS[@]}"; do
    echo "Descargando $archivo..."
    wget -q "https://github.com/ingCesarRomero/edge_ia/raw/main/capitulo3/$archivo"
done


echo "✅ Archivos descargados en la carpeta 'capitulo3'"
ls -la 
