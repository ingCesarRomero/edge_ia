#!/usr/bin/env python3
"""
MÓDULO COMPARTIDO para variables globales entre hilos y archivos
"""

import threading
import numpy as np

# Variables compartidas para todos los archivos
ultimo_frame = None
ultimas_detecciones = []
frame_lock = threading.Lock()
ejecutando = False
