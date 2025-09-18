#!/bin/bash

# ==============================================
# SCRIPT DE DESCARGA DE RECURSOS PARA EDGE IA
# ==============================================

# Colores para mensajes bonitos
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# URLs de los archivos a descargar
URL_MODELO="https://github.com/ingCesarRomero/edge_ia/raw/refs/heads/main/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
URL_ETIQUETAS="https://raw.githubusercontent.com/ingCesarRomero/edge_ia/refs/heads/main/coco_labels.txt"
URL_SCRIPT="https://raw.githubusercontent.com/ingCesarRomero/edge_ia/refs/heads/main/test_mobilenetV2.py"
URL_IMAGEN="https://raw.githubusercontent.com/ingCesarRomero/edge_ia/refs/heads/main/personacaida2.webp"

# Nombres locales de los archivos
ARCHIVO_MODELO="ssd_mobilenet_v2_coco_quant_postprocess.tflite"
ARCHIVO_ETIQUETAS="coco_labels.txt"
ARCHIVO_SCRIPT="test_mobilenetV2.py"
ARCHIVO_IMAGEN="personacaida2.webp"

# Función para mostrar mensajes de estado
mostrar_mensaje() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

mostrar_exito() {
    echo -e "${GREEN}[ÉXITO]${NC} $1"
}

mostrar_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

mostrar_advertencia() {
    echo -e "${YELLOW}[ADVERTENCIA]${NC} $1"
}

# Función para descargar un archivo con verificación
descargar_archivo() {
    local url=$1
    local nombre_archivo=$2
    local descripcion=$3
    
    mostrar_mensaje "Descargando $descripcion..."
    
    if wget --progress=bar:force "$url" -O "$nombre_archivo" 2>&1; then
        if [ -f "$nombre_archivo" ]; then
            tamaño=$(du -h "$nombre_archivo" | cut -f1)
            mostrar_exito "$descripcion descargado correctamente ($tamaño)"
            return 0
        else
            mostrar_error "El archivo $nombre_archivo no se creó después de la descarga"
            return 1
        fi
    else
        mostrar_error "Error al descargar $descripcion desde $url"
        return 1
    fi
}

# Función para verificar dependencias
verificar_dependencias() {
    mostrar_mensaje "Verificando dependencias..."
    
    if ! command -v wget &> /dev/null; then
        mostrar_advertencia "wget no está instalado. Instalando..."
        sudo apt-get update && sudo apt-get install -y wget
    fi
    
    if command -v wget &> /dev/null; then
        mostrar_exito "wget está instalado y listo"
    else
        mostrar_error "No se pudo instalar wget. Instálalo manualmente: sudo apt-get install wget"
        exit 1
    fi
}

# Función principal
main() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  DESCARGADOR DE RECURSOS EDGE IA"
    echo "  Para Raspberry Pi Zero 2W"
    echo "=========================================="
    echo -e "${NC}"
    
    # Verificar que tenemos wget
    verificar_dependencias
    
    echo ""
    mostrar_mensaje "Iniciando descarga de recursos..."
    echo ""
    
    # Descargar todos los archivos
    descargar_archivo "$URL_MODELO" "$ARCHIVO_MODELO" "Modelo MobileNetV2 cuantizado"
    descargar_archivo "$URL_ETIQUETAS" "$ARCHIVO_ETIQUETAS" "Archivo de etiquetas COCO"
    descargar_archivo "$URL_SCRIPT" "$ARCHIVO_SCRIPT" "Script de prueba Python"
    descargar_archivo "$URL_IMAGEN" "$ARCHIVO_IMAGEN" "Imagen de prueba para detección"
    
    echo ""
    echo "=========================================="
    
    # Verificar que todos los archivos se descargaron
    archivos_descargados=0
    archivos_totales=4
    
    [ -f "$ARCHIVO_MODELO" ] && archivos_descargados=$((archivos_descargados + 1))
    [ -f "$ARCHIVO_ETIQUETAS" ] && archivos_descargados=$((archivos_descargados + 1))
    [ -f "$ARCHIVO_SCRIPT" ] && archivos_descargados=$((archivos_descargados + 1))
    [ -f "$ARCHIVO_IMAGEN" ] && archivos_descargados=$((archivos_descargados + 1))
    
    if [ $archivos_descargados -eq $archivos_totales ]; then
        mostrar_exito "¡Todos los archivos se descargaron correctamente!"
        echo ""
        echo "📁 Archivos en el directorio actual:"
        ls -la *.tflite *.txt *.py *.webp 2>/dev/null | awk '{print "   " $9 " (" $5 " bytes)"}'
    else
        mostrar_advertencia "Se descargaron $archivos_descargados de $archivos_totales archivos"
        ls -la *.tflite *.txt *.py *.webp 2>/dev/null
    fi
    
    echo ""
    echo "🚀 Para ejecutar el script de prueba:"
    echo "   python3 $ARCHIVO_SCRIPT"
    echo ""
    echo "📊 Para verificar el modelo:"
    echo "   ls -lh *.tflite && wc -l $ARCHIVO_ETIQUETAS"
    echo ""
}

# Ejecutar función principal
main "$@"
