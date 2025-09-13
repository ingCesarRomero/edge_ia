#!/bin/bash

# crear_venv_ia.sh - Crea un nuevo entorno virtual para IA con OpenCV configurado
# Uso: ./crear_venv_ia.sh [nombre_proyecto] [nombre_entorno]

# Configuración por defecto
PROYECTO=${1:-"mi_proyecto_ia"}
ENTORNO=${2:-"venv"}

RUTA_PROYECTO="$HOME/$PROYECTO"
RUTA_VENV="$RUTA_PROYECTO/$ENTORNO"

echo "🐍 Creando entorno virtual para IA en el Edge..."
echo "📁 Proyecto: $PROYECTO"
echo "🎯 Entorno: $ENTORNO"
echo ""

# Crear directorio del proyecto si no existe
if [ ! -d "$RUTA_PROYECTO" ]; then
    echo "📂 Creando directorio del proyecto: $RUTA_PROYECTO"
    mkdir -p "$RUTA_PROYECTO"
fi

cd "$RUTA_PROYECTO"

# Verificar si el entorno virtual ya existe
if [ -d "$ENTORNO" ]; then
    echo "❌ El entorno virtual '$ENTORNO' ya existe en este proyecto."
    echo "   Por favor, elige otro nombre o elimina el existente."
    exit 1
fi

# Crear el entorno virtual
echo "🛠️ Creando entorno virtual '$ENTORNO'..."
python3 -m venv "$ENTORNO"

# Activar el entorno y instalar paquetes
echo "📦 Instalando paquetes Python..."
source "$ENTORNO/bin/activate"

# Instalar librerías básicas de IA
pip install "numpy<2" pandas tflite-runtime --prefer-binary

# Configurar OpenCV global en el entorno virtual
echo "👁️ Configurando OpenCV..."
SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/usr/lib/python3/dist-packages" >> "$SITE_PACKAGES_PATH/global.pth"

# Verificación final
echo "✅ Verificando la instalación..."
python -c "import numpy as np; print('   NumPy:', np.__version__)"
python -c "import pandas as pd; print('   Pandas:', pd.__version__)"
python -c "import tflite_runtime.interpreter as tflite; print('   TensorFlow Lite: OK')"
python -c "import cv2; print('   OpenCV:', cv2.__version__)"

echo ""
echo "🎉 ¡Entorno virtual creado exitosamente!"
echo ""
echo "📍 Ruta: $RUTA_VENV"
echo "🚀 Para activar:"
echo "   source $RUTA_VENV/bin/activate"
echo ""
echo "🐍 Para desactivar:"
echo "   deactivate"
