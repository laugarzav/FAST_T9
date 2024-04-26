""" Tarea 9: Extracción y comparación de características de imágenes para aplicaciones
de detección de objetos.

Fecha: 26/04/2024

Alumnos: Laura Sofía Garza Villarreal    600650
         Rafael Romero Hurtado           628911
         
"Damos nuestra palabra de que hemos realizado esta actividad con integridad académica"

Ejecución del código: python .\FST.py CI.jpg --threshold 20
"""

# Import standard libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Función para línea de comandos
def parse_arguments():

    """ 
    Analiza los argumentos de la línea de comandos.

    Salida: 
            Namespace: Argumentos analizados"""

    # Analizador de argumentos
    parser = argparse.ArgumentParser(description = "Detect corners in an image.")
    # Argumento para ruta de la imagen
    parser.add_argument('image_path', type = str, help = 'Path to the input image')
    # Argumento para umbral
    parser.add_argument('--threshold', type = int, default = 20, help = 'Threshold for corner detection.')
    return parser.parse_args()

def is_pixel_brighter_than(pixel, threshold):

    """ 
    Función que comprueba si un valor de píxel es más brillante que un umbral determinado
    Argumentos:
        píxel (int): valor de píxel
        umbral (int): valor de umbral

    Salida:
        bool: Verdadero si el pixel es más brillante
              Falso si el pixel es menos brillante
    """
    return pixel > threshold

def is_pixel_darker_than(pixel, threshold):

    """ 
    Función que comprueba si un valor de píxel es más oscuro que un umbral determinado
    Argumentos:
        píxel (int): valor de píxel
        umbral (int): valor de umbral

    Salida:
        bool: Verdadero si el pixel es más oscuro
              Falso si el pixel es menos oscuro
    """
    return pixel < threshold

def test_for_corner(image, center_pixel, pixels, threshold):

    """
    Función que prueba si un píxel es una esquina según las intensidades de los píxeles circundantes.
    Argumentos:
        image(numpy.ndarray): imagen de entrada
        center_pixel(int): valor de intensidad del píxel central
        pixels(lista): lista de intensidades de píxeles circundantes
        threshold(int): valor del umbral

    Salida:
        bool: Verdadero si el píxel es una esquina
              Falso si el píxel no es una esquina
    """
    brighter_count = np.sum(is_pixel_brighter_than(pixels, center_pixel + threshold))
    darker_count = np.sum(is_pixel_darker_than(pixels, center_pixel - threshold))
    return brighter_count >= 9 or darker_count >= 9

def non_maximum_suppression(corner_candidates, distance_threshold):

    """
    Función que realiza una supresión no máxima para filtrar las posibles esquinas
    Argumentos:
        corner_candidates(lista): lista de coordenadas que son posibles esquinas
        distance_threshold(int): umbral para considerar las esquinas como distintas

    Salida:
        list: lista de coordenadas de esquinas filtradas
    """
    corners = []

    for corner in corner_candidates:
        x, y = corner
        is_maximum = True

        for c in corners:
            cx, cy = c
            if abs(x - cx) < distance_threshold and abs(y - cy) < distance_threshold:
                is_maximum = False
                break

        if is_maximum:
            corners.append(corner)

    return corners

def visualize_corners(image, corners):
    
    """
    Función que visualiza las esquinas detectadas en la imagen de entrada
    Argumentos:
        image(numpy.ndarray): imagen de entrada
        corners(lista): lista de coordenadas de esquinas
    """
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        cv2.circle(image_with_corners, (corner[1], corner[0]), 4, (0, 255, 0), -1)

    plt.imshow(image_with_corners)
    plt.show()

def find_corners(image, threshold):

    """
    Función que encuentra las esquinas en la imagen de entrada usando un umbral en específico
    Argumentos:
        image(numpy.ndarray): imagen de entrada
        threshold(int): umbral para la detección de esquinas

    Salidas:
        lista: lista de coordenadas de esquinas
    """
    rows, cols = image.shape
    corner_candidates = []

    for x in range(3, rows - 3):
        for y in range(3, cols - 3):
            center_pixel = image[x, y]
            pixels_around = [
                image[x - 3, y], image[x - 3, y + 1], image[x - 2, y + 2],
                image[x - 1, y + 3], image[x, y + 3], image[x + 1, y + 3],
                image[x + 2, y + 2], image[x + 3, y + 1], image[x + 3, y],
                image[x + 3, y - 1], image[x + 2, y - 2], image[x + 1, y - 3],
                image[x, y - 3], image[x - 1, y - 3], image[x - 2, y - 2],
                image[x - 3, y - 1]
            ]

            if test_for_corner(image, center_pixel, pixels_around, threshold):
                corner_candidates.append((x, y))

    return corner_candidates

def run_pipeline(image_path, threshold):

    """
    Función que ejecuta el proceso de detección de esquinas en la imagen dada
    Argumentos:
        image_path(str): ruta de la imagen de entrada
        threshold(int): umbral para la detección de esquinas
    """

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Encontrar las esquinas usando FAST
    corner_candidates = find_corners(image, threshold)

    # Supresión de no máximos
    final_corners = non_maximum_suppression(corner_candidates, distance_threshold=10)

    # Visualizar resultados
    visualize_corners(image, final_corners)

if __name__ == "__main__":
    # Argumento de ruta de imagen
    args = parse_arguments()

    # Ejecutar el pipeline
    run_pipeline(args.image_path, args.threshold)