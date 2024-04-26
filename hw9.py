import argparse
import cv2

def load_images(img1_path, img2_path):
    """
    Carga dos imágenes desde la ruta especificada.
    
    Args:
        img1_path (str): Ruta de la primera imagen.
        img2_path (str): Ruta de la segunda imagen.
    
    Returns:
        img1 (numpy.ndarray): Matriz de la primera imagen.
        img2 (numpy.ndarray): Matriz de la segunda imagen.
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return img1, img2

def resize_images(img1, img2, width, height):
    """
    Redimensiona dos imágenes para que tengan el mismo tamaño.

    Args:
        img1 (numpy.ndarray): Matriz de la primera imagen.
        img2 (numpy.ndarray): Matriz de la segunda imagen.
        width (int): Ancho deseado para las imágenes redimensionadas.
        height (int): Altura deseada para las imágenes redimensionadas.

    Returns:
        img1_resized (numpy.ndarray): Matriz de la primera imagen redimensionada.
        img2_resized (numpy.ndarray): Matriz de la segunda imagen redimensionada.
    """
    img1_resized = cv2.resize(img1, (width, height))
    img2_resized = cv2.resize(img2, (width, height))
    return img1_resized, img2_resized

def compute_fast_features(img1, img2):
    """
    Detecta características FAST en dos imágenes y calcula los descriptores SIFT correspondientes.

    Args:
        img1 (numpy.ndarray): Matriz de la primera imagen.
        img2 (numpy.ndarray): Matriz de la segunda imagen.

    Returns:
        kp1 (list): Lista de puntos clave detectados en la primera imagen.
        kp2 (list): Lista de puntos clave detectados en la segunda imagen.
        des1 (numpy.ndarray): Descriptores SIFT correspondientes a los puntos clave de la primera imagen.
        des2 (numpy.ndarray): Descriptores SIFT correspondientes a los puntos clave de la segunda imagen.
    """
    # Detector de características FAST
    img_fast = cv2.FastFeatureDetector_create()
    img_fast.setNonmaxSuppression(False)

    # Convertir imágenes a escala de grises
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detectar puntos clave FAST y calcular descriptores SIFT
    kp1 = img_fast.detect(img1_grey, None)
    kp2 = img_fast.detect(img2_grey, None)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.compute(img1_grey, kp1)
    kp2, des2 = sift.compute(img2_grey, kp2)

    return kp1, kp2, des1, des2

def match_keypoints(des1, des2):
    """
    Encuentra correspondencias entre los descriptores SIFT de dos imágenes.

    Args:
        des1 (numpy.ndarray): Descriptores SIFT de la primera imagen.
        des2 (numpy.ndarray): Descriptores SIFT de la segunda imagen.

    Returns:
        matches (list): Lista de correspondencias encontradas entre los descriptores.
    """
    # Crear objeto BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Encontrar correspondencias entre los descriptores
    matches = bf.match(des1, des2)
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    """
    Dibuja las correspondencias entre dos imágenes y las muestra en una ventana.

    Args:
        img1 (numpy.ndarray): Matriz de la primera imagen.
        img2 (numpy.ndarray): Matriz de la segunda imagen.
        kp1 (list): Lista de puntos clave detectados en la primera imagen.
        kp2 (list): Lista de puntos clave detectados en la segunda imagen.
        matches (list): Lista de correspondencias entre los descriptores de las imágenes.
    """
    # Dibujar correspondencias
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar correspondencias
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_rectangle(img, x, y, w, h):
    """
    Dibuja un rectángulo en una imagen dada.

    Args:
        img (numpy.ndarray): Matriz de la imagen en la que se dibujará el rectángulo.
        x (int): Coordenada x del punto superior izquierdo del rectángulo.
        y (int): Coordenada y del punto superior izquierdo del rectángulo.
        w (int): Ancho del rectángulo.
        h (int): Altura del rectángulo.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def cli_interface() -> argparse.Namespace:
    """
    Interfaz de línea de comandos para especificar las rutas de las imágenes de entrada.

    Returns:
        args (argparse.Namespace): Argumentos parseados proporcionados por el usuario.
    """
    # Parsear los parámetros ingresados por el usuario usando la línea de comandos
    parser = argparse.ArgumentParser(description="Merge two images based on feature detection.")
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the first input image')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the second input image')
    args = parser.parse_args()

    # Retornar los argumentos parseados
    return args

if __name__ == "__main__":
    args = cli_interface()
    img1, img2 = load_images(args.input_image1, args.input_image2)
    
    # Redimensionar las imágenes para que quepan en la misma ventana
    width = 700
    height = 600
    img1_resized, img2_resized = resize_images(img1, img2, width, height)
    
    kp1, kp2, des1, des2 = compute_fast_features(img1_resized, img2_resized)
    matches = match_keypoints(des1, des2)

    # Encerrar el objeto repetido en un rectángulo verde
    min_x1, min_y1 = float('inf'), float('inf')
    max_x1, max_y1 = float('-inf'), float('-inf')
    min_x2, min_y2 = float('inf'), float('inf')
    max_x2, max_y2 = float('-inf'), float('-inf')

    for match in matches:
        img1_pt = kp1[match.queryIdx].pt
        img2_pt = kp2[match.trainIdx].pt

        min_x1 = min(min_x1, img1_pt[0])
        min_y1 = min(min_y1, img1_pt[1])
        max_x1 = max(max_x1, img1_pt[0])
        max_y1 = max(max_y1, img1_pt[1])

        min_x2 = min(min_x2, img2_pt[0])
        min_y2 = min(min_y2, img2_pt[1])
        max_x2 = max(max_x2, img2_pt[0])
        max_y2 = max(max_y2, img2_pt[1])

    draw_rectangle(img1_resized, int(min_x1), int(min_y1), int(max_x1 - min_x1), int(max_y1 - min_y1))
    draw_rectangle(img2_resized, int(min_x2), int(min_y2), int(max_x2 - min_x2), int(max_y2 - min_y2))

    # Dibujar las correspondencias en ambas imágenes
    draw_matches(img1_resized, img2_resized, kp1, kp2, matches)
