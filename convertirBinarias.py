import json
import numpy as np
from PIL import Image
import cv2

from pathlib import Path

def json_a_mascara(json_file, imagen_size=(512, 512)):
    """
    Convierte una etiqueta JSON de LabelMe a una máscara binaria.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Crear una imagen en blanco (negra) con el tamaño de la imagen original
    mascara = np.zeros(imagen_size, dtype=np.uint8)
    
    # Iterar sobre las formas en el archivo JSON y crear la máscara
    for shape in data['shapes']:
        # Los puntos de la forma
        puntos = np.array(shape['points'], dtype=np.int32)
        
        # Dibujar la forma en la máscara (rellenar la región)
        cv2.fillPoly(mascara, [puntos], 255)  # 255 es el valor blanco

    # Asegurar que la carpeta 'ImagenesBinarias' exista
    output_dir = Path('./ImagenesBinarias')
    output_dir.mkdir(exist_ok=True)

    # Construir el nombre del archivo de salida
    nombre_mascara = output_dir / f"{Path(json_file).stem}.png"

    # Guardar la máscara como una imagen PNG
    Image.fromarray(mascara).save(nombre_mascara)

    return nombre_mascara

# Ejemplo de uso
for i in range(1, 31):
    json_a_mascara(f'./etiquetas/Mastocito{i}.json')
