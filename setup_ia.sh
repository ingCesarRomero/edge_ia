#!/bin/bash

# setup_ia_edge.sh - Script de configuración automática para IA en el Edge en Raspberry Pi
# Uso: ./setup_ia_edge.sh [nombre_proyecto] [nombre_entorno]
# Ejemplo: ./setup_ia_edge.sh mi_ia_project venv_ia

# --- Configuración por defecto ---
DEFAULT_PROJECT_NAME="mi_proyecto_ia"
DEFAULT_VENV_NAME="venv"

# --- Obtener parámetros o usar valores por defecto ---
PROJECT_NAME=${1:-$DEFAULT_PROJECT_NAME}
VENV_NAME=${2:-$DEFAULT_VENV_NAME}

echo "=================================================="
echo "    Configuración Automática para IA en el Edge"
echo "=================================================="
echo ""
echo "📁 Proyecto: $PROJECT_NAME"
echo "🐍 Entorno virtual: $VENV_NAME"
echo ""

# --- Paso 1: Actualizar el sistema e instalar dependencias del sistema ---
echo "[1/5] Actualizando sistema e instalando dependencias base..."
sudo apt update

sudo apt install -y libopenblas-dev python3-venv python3-pip python3-opencv

# --- Paso 2: Crear directorio de proyecto y entorno virtual ---
echo "[2/5] Creando entorno virtual '$VENV_NAME' en proyecto '$PROJECT_NAME'..."
mkdir -p ~/"$PROJECT_NAME"
cd ~/"$PROJECT_NAME"
python3 -m venv "$VENV_NAME"

# --- Paso 3: Instalar paquetes Python en el entorno virtual ---
echo "[3/5] Instalando paquetes Python en el entorno virtual '$VENV_NAME'..."
source ~/"$PROJECT_NAME"/"$VENV_NAME"/bin/activate

# Instalar numpy compatible (versión 1.x)
pip install "numpy<2" --force-reinstall

# Instalar otras dependencias esenciales
pip install pandas tflite-runtime --prefer-binary

# --- Paso 4: Configurar OpenCV global en el entorno virtual ---
echo "[4/5] Configurando OpenCV para el entorno virtual..."

# Encontrar la ruta exacta de site-packages y crear archivo .pth
SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/usr/lib/python3/dist-packages" >> "$SITE_PACKAGES_PATH/global.pth"

# --- Paso 5: Verificación final ---
echo "[5/5] Realizando verificación final..."
echo ""
echo "--- Verificando instalaciones ---"

# Verificar numpy
python -c "import numpy as np; print('✅ NumPy version:', np.__version__)"

# Verificar pandas
python -c "import pandas as pd; print('✅ Pandas version:', pd.__version__)"

# Verificar TensorFlow Lite
python -c "import tflite_runtime.interpreter as tflite; print('✅ TensorFlow Lite: OK')"

# Verificar OpenCV
python -c "import cv2; print('✅ OpenCV version:', cv2.__version__)"

echo ""
echo "=================================================="
echo "    Configuración completada exitosamente!"
echo ""
echo "📊 Resumen de la instalación:"
echo "  📂 Proyecto: ~/$PROJECT_NAME"
echo "  🐍 Entorno virtual: $VENV_NAME"
echo "  📦 NumPy $(python -c 'import numpy as np; print(np.__version__)')"
echo "  👁️ OpenCV $(python -c 'import cv2; print(cv2.__version__)')"
echo "  🤖 TensorFlow Lite Runtime"
echo "  🗃️ Pandas"
echo ""
echo "🎯 Para activar el entorno virtual:"
echo "  source ~/$PROJECT_NAME/$VENV_NAME/bin/activate"
echo ""
echo "¡Tu entorno de IA en el Edge está listo!"
echo "=================================================="