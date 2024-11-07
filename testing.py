import os
import pydicom
import numpy as np
from skimage import segmentation, color, filters, measure
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import cv2
import random
from skimage.segmentation import active_contour, morphological_chan_vese, checkerboard_level_set
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Cargar archivos DICOM
def load_dicom_files(folder):
    dicom_files = []
    for filename in os.listdir(folder):
        if filename.endswith('.dcm'):
            try:
                dicom_files.append(pydicom.dcmread(os.path.join(folder, filename)))
            except Exception as e:
                print(f"Error al leer el archivo {filename}: {e}")
    return dicom_files

# Asegurar el orden correcto de las imágenes
def sort_slices(dicom_files):
    try:
        return sorted(dicom_files, key=lambda s: float(getattr(s, 'SliceLocation', s.ImagePositionPatient[2])))
    except AttributeError as e:
        print(f"Error al ordenar las slices: {e}")
        raise

# Crear una imagen 3D de los cortes y extraer posiciones Z
def create_3d_image(slices):
    img_shape = list(slices[0].pixel_array.shape)
    num_slices = len(slices)
    img_shape.append(num_slices)
    img3d = np.zeros(img_shape, dtype=np.float32)
    z_positions = []

    for i, s in enumerate(slices):
        try:
            img2d = s.pixel_array.astype(np.float32)
            if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
                img2d = img2d * float(s.RescaleSlope) + float(s.RescaleIntercept)
            img3d[:, :, i] = img2d

            if hasattr(s, 'ImagePositionPatient'):
                z = float(s.ImagePositionPatient[2])
            elif hasattr(s, 'SliceLocation'):
                z = float(s.SliceLocation)
            else:
                z = i
                print(f"Advertencia: Slice {i} no tiene información de posición Z. Usando índice como Z.")
            z_positions.append(z)
        except Exception as e:
            print(f"Error al procesar slice {i}: {e}")
            z_positions.append(i)
    return img3d, img_shape, z_positions

# Obtener el corte en el plano especificado
def slice_det(image, index, plane):
    if plane == 0:  # Coronal
        return image[index, :, :]
    elif plane == 1:  # Sagital
        return image[:, index, :]
    elif plane == 2:  # Axial
        return image[:, :, index]

def mostrar_histograma(imagen, titulo="Histograma de Intensidades"):
    plt.figure()
    plt.hist(imagen.ravel(), bins=256, color='gray', alpha=0.7, range=(0, 255))
    plt.title(titulo)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.show()

## Función para segmentar automáticamente una slice específica
#def segmentar_slice_automatica(slice_img):
#
#    # Paso 1: Filtrado y suavizado
#    img_suavizada = cv2.GaussianBlur(slice_img, (5,5), 0.5)
#    
#    # Paso 1: Filtrado de la imagen para suavizar ruido
#    imagen_suavizada = cv2.GaussianBlur(slice_img, (5,5), 0.5)
#    
#    # Paso 2: Detección de bordes con Sobel para obtener contornos
#    gradiente_x = filters.sobel_h(imagen_suavizada)
#    gradiente_y = filters.sobel_v(imagen_suavizada)
#    bordes = np.hypot(gradiente_x, gradiente_y)
#    
#    # Paso 3: Binarización de la imagen de bordes
#    _, umbral = cv2.threshold(bordes, 33, 65, cv2.THRESH_BINARY)
#    bordes_binarios = bordes > umbral
#    
#    # Paso 4: Crecimiento de regiones basado en conectividad
#    # Inicia el crecimiento de la región desde un punto semilla (adaptar posición)
#    semillas = np.zeros_like(bordes_binarios)
#    semillas[325, 268] = 1  # Coordenada inicial ejemplo, ajustar según región
#    
#    # Expandir la región con conectividad 8
#    regiones = morphology.binary_dilation(semillas, morphology.square(3))
#    for _ in range(5):  # Realizar varias iteraciones para crecimiento
#        regiones = morphology.binary_dilation(regiones)
#    
#    # Paso 5: Contornos Activos (Snakes) para ajustarse mejor a los bordes de la vértebra
#    snake = morphology.binary_erosion(regiones)  # Iniciar contorno cercano
#    for _ in range(50):  # Número de iteraciones de ajuste
#        grad_ext = filters.sobel(bordes_binarios.astype(float))
#        fuerza_externa = grad_ext - cv2.GaussianBlur(slice_img, (5,5), 1)
#        snake = snake + fuerza_externa * 0.1  # Ajuste ligero del contorno
#    
#    # Paso 6: Extraer etiquetas de regiones conectadas
#    etiquetas = measure.label(snake < 0.5)
#    
#    return etiquetas

# Función para segmentar automáticamente una slice específica

def segmentar_slice_automatica(ct_slice, initial_threshold=112, max_iter=10, window_size=5):
    """
    Segmentación de hueso en un slice 2D de una imagen de CT utilizando umbralización adaptativa en 2D.

    Args:
    ct_slice (numpy.ndarray): Imagen 2D de un slice de CT.
    initial_threshold (float): Umbral inicial para separar hueso y no-hueso.
    max_iter (int): Número máximo de iteraciones.
    window_size (int): Tamaño de la ventana para el ajuste local.

    Returns:
    numpy.ndarray: Máscara binaria 2D con la región segmentada de hueso.
    """
    # Paso 1: Umbral inicial
    binary_image = ct_slice >= initial_threshold

    # Paso 2: Iteración adaptativa
    for _ in range(max_iter):
        # Detectar los bordes de la región actual de hueso
        edges = ndimage.binary_dilation(binary_image) & ~binary_image
        
        # Ajustar umbrales localmente en las regiones de borde
        for (y, x) in zip(*np.where(edges)):
            # Definir la ventana alrededor del píxel
            y_min, y_max = max(0, y - window_size), min(ct_slice.shape[0], y + window_size + 1)
            x_min, x_max = max(0, x - window_size), min(ct_slice.shape[1], x + window_size + 1)
            
            # Extraer la región de la ventana y calcular su media
            local_region = ct_slice[y_min:y_max, x_min:x_max]
            mean_intensity = np.mean(local_region)
            
            # Reajustar el píxel en base a la intensidad local
            if ct_slice[y, x] >= mean_intensity:
                binary_image[y, x] = True
            else:
                binary_image[y, x] = False

    # Paso 3: Post-procesamiento con "region growing"
    labels, _ = ndimage.label(binary_image)
    largest_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  # Mantener la mayor región conectada

    return largest_component.astype(np.uint8)

def binarization_based_segmentation(slice_img):

    # Paso 1: Filtrado y suavizado
    img_suavizada = cv2.GaussianBlur(slice_img, (5,5), 0.5)
    
    mostrar_histograma(img_suavizada, "Histograma de la Imagen Suavizada")

    # Paso 2: Detección de bordes con Sobel
    grad_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 3, 0, ksize=5)
    grad_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 3, ksize=5)
    bordes = np.hypot(grad_x, grad_y)
    
    # Normalización de bordes para convertir a uint8
    bordes = (bordes / bordes.max() * 255).astype(np.uint8)
    
    # Paso 3: Binarización de la imagen de bordes
    _, bordes_binarios1 = cv2.threshold(bordes, 33, 70, cv2.THRESH_BINARY)
    
    test = cv2.Laplacian(bordes_binarios1, cv2.CV_64F, ksize=5)
    bordes1 = (test / test.max() * 255).astype(np.uint8)

    _, bordes_binarios = cv2.threshold(bordes1, 0, 255, cv2.THRESH_OTSU)
    # Paso 4: Aplicar crecimiento de regiones
    semilla = (324, 330)
    mask = np.zeros((slice_img.shape[0] + 2, slice_img.shape[1] + 2), np.uint8)  # Crear máscara para floodFill
    cv2.floodFill(bordes_binarios1, mask, semilla, (33, 70, 255), flags=cv2.FLOODFILL_MASK_ONLY)
    
    return mask[1:-1, 1:-1]  # Retornar la máscara sin los bordes adicionales

def segmentacion_mumford_shah(slice_img, iteraciones=200, lambda1=1, lambda2=1):
    # Preprocesamiento: Suavizar la imagen para reducir el ruido
    img_suavizada = cv2.GaussianBlur(slice_img, (5,5), 0.5)

    # Inicializar el nivel de conjunto inicial (checkerboard)
    init_level_set = checkerboard_level_set(slice_img.shape, 6)

    # Aplicar la segmentación de Mumford-Shah (aproximada por Chan-Vese morfológica)
    segmentacion = morphological_chan_vese(
        img_suavizada, 
        num_iter=iteraciones, 
        init_level_set=init_level_set, 
        smoothing=1
    )

    # Visualizar la segmentación
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(slice_img, cmap='gray')
    ax[0].set_title("Imagen Original")
    ax[0].axis('off')
    
    ax[1].imshow(slice_img, cmap='gray')
    ax[1].contour(segmentacion, [0.5], colors='r')  # Dibujar el contorno en la imagen
    ax[1].set_title("Segmentación con Mumford-Shah (Chan-Vese)")
    ax[1].axis('off')
    
    plt.show()

    return segmentacion


def ajustar_contorno_activo(slice_img, mask):
    # Crear un contorno inicial (ej. un círculo alrededor de la región de interés)
    s = np.linspace(0, 2 * np.pi, 100)
    #r = 320 + 25 * np.sin(s)  # Radio inicial
    #c = 330 + 25 * np.cos(s)  # Centro del círculo
    r = 256 + 25 * np.sin(s)  # Radio inicial coronal
    c = 360 + 25 * np.cos(s)  # Centro del círculo coronal

    
    init_contour = np.array([r, c]).T

    # Ajustar el contorno activo a los bordes detectados
    contorno_final = active_contour(
        cv2.GaussianBlur(slice_img, (5,5), 0.5),
        init_contour,
        alpha=0.01,  # Peso de suavidad del contorno
        beta=10,   # Peso de rigidez del contorno
        gamma= 1   # Tasa de actualización de iteración
    )

    # Mostrar el contorno final sobre la imagen original
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')
    ax.plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=1)  # Contorno inicial
    ax.plot(contorno_final[:, 1], contorno_final[:, 0], '-b', lw=2)  # Contorno ajustado
    ax.set_title("Contorno Activo Ajustado")
    plt.show()

    return contorno_final


def ajustar_contorno_activo_automatico(slice_img, pixel_spacing):
    # Preprocesamiento de imagen: detección de bordes y umbral adaptativo
    edges = sobel(slice_img)  # Usar sobel para detectar bordes de las vértebras
    thresh_val = threshold_otsu(edges)  # Umbral de Otsu para separar fondo y vértebras
    binary_mask = edges > thresh_val  # Crear una máscara binaria
    
    # Etiquetado de regiones conectadas para obtener cada vértebra por separado
    num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    
    vertebra_contours = []
    for label in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        # Crear una máscara para cada vértebra
        vertebra_mask = (labels == label).astype(np.uint8)
        
        # Calcular contorno inicial de forma automática alrededor de la región etiquetada
        contours, _ = cv2.findContours(vertebra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = contours[0].squeeze()
        
        # Si el contorno es demasiado pequeño, ignorarlo
        if cnt.shape[0] < 5:
            continue

        # Aplicar el contorno activo usando el contorno inicial
        init_contour = cnt[:, [1, 0]]  # Cambiar a formato (fila, columna) para `active_contour`
        
        # Ajustar el contorno activo a los bordes detectados
        contorno_final = active_contour(
            cv2.GaussianBlur(slice_img, (5, 5), 0.5),
            init_contour,
            alpha=0.01,  # Peso de suavidad del contorno
            beta=10,     # Peso de rigidez del contorno
            gamma=0.1,   # Tasa de actualización de iteración
            max_iterations=2500,  # Incrementar iteraciones si es necesario
        )
        
        vertebra_contours.append(contorno_final)
        
        # Mostrar el contorno final sobre la imagen original
        fig, ax = plt.subplots()
        ax.imshow(slice_img, cmap='gray')
        ax.plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=1)  # Contorno inicial
        ax.plot(contorno_final[:, 1], contorno_final[:, 0], '-b', lw=2)  # Contorno ajustado
        ax.set_title(f"Contorno Activo Ajustado - Vértebra {label}")
        plt.show()
        
    return vertebra_contours

def adaptive_thresholding_3d(ct_image, initial_threshold, max_iter=10, window_size=3, slice_range=None):
    """
    Segmentación de hueso en un volumen de imágenes de CT utilizando umbralización adaptativa en 3D.
    
    Args:
    ct_image (numpy.ndarray): Volumen de imágenes de CT 3D.
    initial_threshold (float): Umbral inicial para separar hueso y no-hueso.
    max_iter (int): Número máximo de iteraciones.
    window_size (int): Tamaño de la ventana para el ajuste local.
    slice_range (tuple): Rango de cortes en el eje z, como (inicio, fin). Si es None, procesa todo el volumen.

    Returns:
    numpy.ndarray: Máscara binaria 3D con la región segmentada de hueso.
    """
    # Limitar el volumen al rango de cortes especificado
    if slice_range:
        z_start, z_end = slice_range
        ct_image = ct_image[z_start:z_end]

    # Paso 1: Umbral inicial
    binary_image = ct_image >= initial_threshold

    # Paso 2: Iteración adaptativa
    for _ in range(max_iter):
        # Detectar los bordes de la región actual de hueso
        edges = ndimage.binary_dilation(binary_image) & ~binary_image
        
        # Ajustar umbrales localmente en las regiones de borde
        for (z, y, x) in zip(*np.where(edges)):
            # Definir la ventana alrededor del voxel
            z_min, z_max = max(0, z - window_size), min(ct_image.shape[0], z + window_size + 1)
            y_min, y_max = max(0, y - window_size), min(ct_image.shape[1], y + window_size + 1)
            x_min, x_max = max(0, x - window_size), min(ct_image.shape[2], x + window_size + 1)
            
            # Extraer la región de la ventana y calcular su media
            local_region = ct_image[z_min:z_max, y_min:y_max, x_min:x_max]
            mean_intensity = np.mean(local_region)
            
            # Reajustar el voxel en base a la intensidad local
            if ct_image[z, y, x] >= mean_intensity:
                binary_image[z, y, x] = True
            else:
                binary_image[z, y, x] = False

    # Paso 3: Post-procesamiento con "region growing"
    labels, _ = ndimage.label(binary_image)
    largest_component = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  # Mantener la mayor región conectada

    return largest_component.astype(np.uint8)


# Función para generar la superficie B-spline utilizando interpolación cúbica
#def generate_b_spline_surface(points_3d, resolution=100):
#    """
#    Genera una superficie B-spline a partir de puntos 3D utilizando interpolación cúbica.
#    
#    Args:
#        points_3d (ndarray): Puntos 3D de la vértebra (N, 3).
#        resolution (int): Resolución de la grilla.
#    
#    Returns:
#        tuple: (grid_x, grid_y, grid_z)
#    """
#    # Separar las coordenadas x, y y z de los puntos 3D
#    x = points_3d[:, 0]
#    y = points_3d[:, 1]
#    z = points_3d[:, 2]
#    
#    # Crear una grilla para los puntos xy
#    grid_x, grid_y = np.mgrid[min(x):max(x):complex(resolution), min(y):max(y):complex(resolution)]
#    
#    # Interpolación B-Spline para obtener valores de z en la grilla
#    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
#    
#    # Reemplazar NaN con interpolación 'nearest'
#    mask_nan = np.isnan(grid_z)
#    if np.any(mask_nan):
#        grid_z[mask_nan] = griddata((x, y), z, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')
#    
#    return grid_x, grid_y, grid_z

## Crear máscara desde la spline
#def create_mask_from_spline(spline_x, spline_y, img_shape):
#    spline_path = Path(np.column_stack((spline_x, spline_y)))
#    x, y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
#    points = np.column_stack((x.ravel(), y.ravel()))
#    mask = spline_path.contains_points(points)
#    mask = mask.reshape(img_shape)
#    return mask

# Función para medir altura considerando la dirección principal de la vértebra
#def measure_heights_principal(mask, pixel_spacing):
#    # Extraer los contornos de la máscara
#    contours = measure.find_contours(mask, 0.5)
#    
#    if not contours:
#        print("No se encontraron contornos en la máscara.")
#        return None, None
#    
#    # Seleccionar el contorno más grande
#    contour = max(contours, key=len)
#    
#    # Convertir las coordenadas del contorno a puntos (x, y)
#    points = np.fliplr(contour)  # flip to (x, y)
#    
#    # Aplicar PCA para encontrar la dirección principal
#    pca = PCA(n_components=2)
#    pca.fit(points)
#    
#    principal_axis = pca.components_[0]
#    secondary_axis = pca.components_[1]
#    
#    # Proyectar los puntos sobre la dirección principal y secundaria
#    projections = pca.transform(points)
#    
#    # Calcular la altura como la diferencia máxima en la dirección principal
#    height = projections[:, 0].max() - projections[:, 0].min()
#    
#    # Convertir la altura a milímetros
#    height_mm = height * pixel_spacing[0]  # Asumiendo que principal axis está alineado con y
#    
#    # Crear una imagen de alturas por píxel (simplemente un ejemplo)
#    height_image_mm = projections[:, 0].reshape(mask.shape) * pixel_spacing[0]
#    
#    return height_image_mm, height_mm


## Visualización de las alturas en color o escala de grises
#def plot_height_images_principal(height_image_mm, pixel_spacing):
#    """
#    Visualiza la imagen de alturas utilizando un mapa de colores.
#    
#    Args:
#        height_image_mm (ndarray): Imagen 2D de alturas en mm.
#        pixel_spacing (tuple): Espaciado de píxeles en (y, x, z) en mm.
#    """
#    if height_image_mm is None:
#        print("No se pudo generar la imagen de alturas.")
#        return
#    
#    # Reemplazar NaN y Inf con 0
#    height_image_mm = np.nan_to_num(height_image_mm, nan=0.0, posinf=0.0, neginf=0.0)
#    
#    # Obtener dimensiones
#    height, width = height_image_mm.shape
#    
#    # Crear ejes físicos
#    extent = [
#        0, width * pixel_spacing[1],  # X-axis
#        0, height * pixel_spacing[0] # Y-axis
#    ]
#    
#    plt.figure(figsize=(8, 6))
#    im = plt.imshow(height_image_mm, cmap='gray', extent=extent, origin='lower', aspect='auto')
#    
#    # Añadir colorbar y ajustar los ticks para dividir los valores por 10
#    cbar = plt.colorbar(im, label='Altura (mm)')
#    
#    # Obtener los ticks actuales de la colorbar
#    ticks = cbar.get_ticks()
#    
#    # Modificar los ticks para dividir por 10
#    cbar.set_ticks(ticks)
#    cbar.set_ticklabels(ticks / 10)
#    
#    plt.title("Altura en Dirección Principal")
#    plt.xlabel("Distancia Principal (mm)")
#    plt.ylabel("Distancia Secundaria (mm)")
#    plt.tight_layout()
#    plt.show()
#
## Graficar la superficie B-spline
#def plot_3d_surface(grid_x, grid_y, grid_z):
#    fig = plt.figure(figsize=(10, 7))
#    ax = fig.add_subplot(111, projection='3d')
#    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='gray', edgecolor='none')
#    ax.set_xlabel('X (Izquierda-Derecha) [mm]')
#    ax.set_ylabel('Y (Anteroposterior) [mm]')
#    ax.set_zlabel('Z (Altura) [mm]')
#    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Altura (mm)')
#    plt.show()
#
#
#def plot_height_images_lr_ap(grid_x, grid_y, grid_z, pixel_spacing):
#    # Convertir las coordenadas a unidades físicas (mm)
#    grid_x_mm = grid_x * pixel_spacing[0]
#    grid_y_mm = grid_y * pixel_spacing[1]
#    grid_z_mm = grid_z * pixel_spacing[2]
#
#    plt.figure(figsize=(12, 6))
#
#    # Superficie Izquierda-Derecha (X vs Z), mostrando la variación en Y
#    plt.subplot(1, 2, 1)
#    cont1 = plt.contourf(grid_x_mm, grid_z_mm, grid_y_mm, cmap='gray')
#    cbar1 = plt.colorbar(cont1, label='Distancia Anteroposterior (Y) (mm)')
#    
#    # Ajustar los valores de la colorbar para dividir por 10
#    ticks = cbar1.get_ticks()
#    cbar1.set_ticks(ticks)
#    cbar1.set_ticklabels(ticks / 10)  # Dividir los valores por 10
#    
#    plt.title("Superficie Izquierda-Derecha (X vs Z)")
#    plt.xlabel("Distancia Izquierda-Derecha (mm)")
#    plt.ylabel("Altura (Z) (mm)")
#
#    # Superficie Anteroposterior (X vs Y), mostrando la variación en Z
#    plt.subplot(1, 2, 2)
#    cont2 = plt.contourf(grid_x_mm, grid_y_mm, grid_z_mm, cmap='gray')
#    cbar2 = plt.colorbar(cont2, label='Altura (Z) (mm)')
#    
#    # Ajustar los valores de la colorbar para dividir por 10
#    ticks2 = cbar2.get_ticks()
#    cbar2.set_ticks(ticks2)
#    cbar2.set_ticklabels(ticks2 / 10)  # Dividir los valores por 10
#    
#    plt.title("Superficie Anteroposterior (X vs Y)")
#    plt.xlabel("Distancia Izquierda-Derecha (mm)")
#    plt.ylabel("Distancia Anteroposterior (Y) (mm)")
#
#    plt.tight_layout()
#    plt.show()

# Función para medir la altura en la dirección pies-cabeza (eje vertical) de cada vértebra segmentada
def measure_heights_lr(mask, pixel_spacing):
    labels = measure.label(mask)
    height_map = np.zeros_like(mask, dtype=np.float32)
    
    for region_label in np.unique(labels):
        if region_label == 0:  # Ignorar el fondo
            continue
            
        # Crear una máscara para cada vértebra segmentada
        region_mask = (labels == region_label)
        
        # Calcular las coordenadas de los puntos de borde
        y_coords, x_coords = np.where(region_mask)
        
        # Calcular la altura en la dirección pies-cabeza (eje vertical)
        height = (y_coords.max() - y_coords.min()) * pixel_spacing[0]
        
        # Asignar el valor de altura en el mapa de altura
        height_map[region_mask] = height
    
    return height_map

# Función para medir la altura en la dirección principal de cada vértebra segmentada usando PCA
def measure_heights_pca(mask, pixel_spacing):
    labels = measure.label(mask)
    height_map_pca = np.zeros_like(mask, dtype=np.float32)
    
    for region_label in np.unique(labels):
        if region_label == 0:  # Ignorar el fondo
            continue
            
        # Crear una máscara para cada vértebra segmentada
        region_mask = (labels == region_label)
        
        # Calcular las coordenadas de los puntos de borde
        y_coords, x_coords = np.where(region_mask)
        points = np.column_stack((x_coords, y_coords))
        
        # Aplicar PCA para encontrar la dirección principal
        pca = PCA(n_components=2)
        pca.fit(points)
        projections = pca.transform(points)
        
        # Calcular la altura en la dirección principal
        height_pca = (projections[:, 0].max() - projections[:, 0].min()) * pixel_spacing[0]
        
        # Asignar el valor de altura en el mapa de altura PCA
        height_map_pca[region_mask] = height_pca
    
    return height_map_pca

# Función para graficar las alturas como imágenes con escala de grises o código de color
def plot_height_maps(height_map, height_map_pca):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Grafico de alturas en la dirección pies-cabeza
    im1 = axes[0].imshow(height_map, cmap='gray')
    axes[0].set_title("Altura en dirección pies-cabeza")
    fig.colorbar(im1, ax=axes[0], label="Altura (mm)")
    
    # Grafico de alturas en la dirección principal (PCA)
    im2 = axes[1].imshow(height_map_pca, cmap='gray')
    axes[1].set_title("Altura en dirección principal (PCA)")
    fig.colorbar(im2, ax=axes[1], label="Altura (mm)")
    
    plt.show()


    
# Mostrar cortes coronales
def show_coronal_slices(img3d, img_shape, z_positions, pixel_spacing):
    plane = 1  # Sagital
    max_index = img_shape[plane] - 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    initial_index = max_index // 2
    img_display = slice_det(img3d, initial_index, plane)
    img_plot = ax.imshow(img_display, cmap='gray')
    ax.set_title(f"Corte Coronal: {initial_index}")

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax=ax_slider, label='Corte Coronal', valmin=0, valmax=max_index, valinit=initial_index, valstep=1, color='blue')

    vertebrae = []
    current_vertebra = {}
    markers = {}
    colors_cmap = plt.get_cmap('tab10', 10)
    vertebra_count = 0
    bspline_surfaces = []

    def update(val):
        index = int(slider.val)
        img = slice_det(img3d, index, plane)
        img_plot.set_data(img)
        ax.set_title(f"Corte Coronal: {index}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    slice_img = None
    mask = None
    selected_slice = None
    vertebrae_extraction_normalized = None
    closed = None
    def on_key(event):
        nonlocal slice_img, mask, selected_slice, vertebrae_extraction_normalized, closed
        if event.key == 'x':
            segmented_3d = adaptive_thresholding_3d(img3d, initial_threshold=112, slice_range =[256, 266])
            # Aplicar el algoritmo de Marching Cubes para extraer la superficie
            verts, faces, _, _ = measure.marching_cubes(segmented_3d, level=0)

            # Visualizar la superficie en 3D
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Crear la colección de polígonos 3D para la superficie
            mesh = Poly3DCollection(verts[faces], alpha=0.1, edgecolor='k')
            ax.add_collection3d(mesh)

            # Configurar límites y etiquetas de los ejes
            ax.set_xlim(0, segmented_3d.shape[0])
            ax.set_ylim(0, segmented_3d.shape[1])
            ax.set_zlim(0, segmented_3d.shape[2])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            plt.show()
        elif event.key == 'r':
            selected_slice = int(slider.val)
            slice_img = slice_det(img3d, selected_slice, plane)
            mask = segmentar_slice_automatica(slice_img)
            plt.figure()
            plt.imshow(mask, cmap='gray')
            plt.title(f"Segmentación automática - Slice {selected_slice}")
            plt.show()
        elif event.key == 'b':
            contorno_final = segmentacion_mumford_shah(mask)
            #vertebrae_extraction = (contorno_final)*(1-mask)
            filled_image = np.uint8(contorno_final.copy())
            # Definir el kernel para la operación morfológica
            kernel = np.ones((3,3), np.uint8)  # Puedes ajustar el tamaño según los huecos
#            # Aplicar operaciones de dilatación y cierre (closing)
            dilated = cv2.dilate(filled_image, kernel, iterations=1)
            closed = cv2.erode(dilated, kernel, iterations=1)
#            # Mostrar el resultado
            plt.figure()
            plt.imshow(closed, cmap='gray')
            plt.title("Imagen con Operación de Cierre")
            plt.show()

            vertebrae_extraction = (slice_img) * (closed)
            vertebrae_extraction_normalized = cv2.normalize(vertebrae_extraction, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #threshold 100 y 110 en sagital
            #threshold 140 y 150 en coronal
            # MEJOR DE TODAS ES LA SLICE 341 CORONAL
            # MEJOR SLICE 269 SAGITAL
            plt.figure()
            plt.imshow(vertebrae_extraction_normalized, cmap='gray')
            plt.title(f"Vértebras Extraída con Contorno - Slice {selected_slice}")
            plt.show()

            #mostrar_histograma(vertebrae_extraction_normalized, "Histograma de la Vértebra Extraída")
            #_, vertebrae_extraction_normalized_tst = cv2.threshold(vertebrae_extraction_normalized, 100, 110, cv2.THRESH_BINARY)
#
            #plt.figure()
            #plt.imshow(vertebrae_extraction_normalized_tst, cmap='gray')
            #plt.title(f"Vértebras Segmentadas - Slice {selected_slice}")
            #plt.show()
#
#
            #height, width = vertebrae_extraction_normalized_tst.shape
            ### ROI PARA CORONAL
            ##roi = vertebrae_extraction_normalized_tst[int(height * 0.3):int(height * 0.7), :int(width * 0.9)]
#
            ### ROI PARA SAGITAL
            #roi = vertebrae_extraction_normalized_tst[int(height * 0.5):int(height * 0.9), :int(width * 0.9)]
#
            #            # Aplicar la transformada de distancia para identificar el centro de las regiones
            #dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
#
            ## Normalizar la imagen de distancia para que esté en el rango 
            #cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
#
            #plt.figure()
            #plt.title("Imagen de Distancia Normalizada")
            #plt.imshow(dist, cmap='gray')
            #plt.show()
#
            #mostrar_histograma(dist, "Histograma de la Imagen de Distancia")
            ## Aplicar un umbral para obtener los picos, que serán los marcadores para los objetos de primer plano
            #_, dist_bin  = cv2.threshold(dist, 24, 100, cv2.THRESH_BINARY)
#
            ## Convertir la imagen de distancia a tipo uint8 para encontrar contornos
            #dist_8u = dist_bin.astype(np.uint8)
#
            ## Encontrar contornos en la imagen umbralizada
            #contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
            ## Crear la imagen de marcadores para el algoritmo watershed
            #markers = np.zeros(dist.shape, dtype=np.int32)
#
            ## Dibujar los marcadores de los objetos de primer plano
            #for i in range(len(contours)):
            #    cv2.drawContours(markers, contours, i, i + 1, -1)  # Los índices empiezan en 1
#
            ## Dibujar un marcador para el fondo (parte superior izquierda de la imagen)
            #cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
#
            ## Visualizar los marcadores antes de aplicar watershed (opcional)
            #markers_8u = (markers * 10).astype('uint8')
            #plt.figure()
            #plt.title("Markers")
            #plt.imshow(markers_8u, cmap='gray')
            #plt.show()
#
            ## Aplicar el algoritmo de watershed
            #imgResult = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convertir la imagen a color para watershed
            #cv2.watershed(imgResult, markers)
#
            ## Convertir los marcadores a uint8 y hacer una inversión para visualizar (opcional)
            #mark = markers.astype('uint8')
            #mark = cv2.bitwise_not(mark)
#
            ## Generar colores aleatorios para cada región detectada
            #random.seed(42)
            #colors = []
            #for contour in contours:
            #    colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
#
            ## Crear la imagen de resultado
            #dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
#
            ## Rellenar los objetos etiquetados con colores aleatorios
            #for i in range(markers.shape[0]):
            #    for j in range(markers.shape[1]):
            #        index = markers[i, j]
            #        if index > 0 and index <= len(contours):
            #            dst[i, j, :] = colors[index - 1]
#
            ## Mostrar la imagen final con los objetos coloreados
            #plt.figure()
            #plt.title("Final Result")
            #plt.imshow(dst, cmap='gray')
            #plt.show()
#
            ## Dilatar la imagen de distancia para resaltar los picos (opcional)
            #kernel1 = np.ones((3, 3), np.uint8)
            #dist = cv2.dilate(dist, kernel1)
            #closed = cv2.erode(dist, kernel1, iterations=2)
            #plt.figure()
            #plt.title("Peaks")
            #plt.imshow(closed, cmap='gray')
            #plt.show()
            #
#
            # Crear una copia de la imagen binaria para el relleno
        #    filled_image = roi.copy()
        #    
        #    # Encontrar las áreas de los contornos y rellenarlas usando flood fill
        #    height, width = filled_image.shape
        #    for y in range(0, height):
        #        for x in range(0, width):
        #            # Iniciar el relleno en cada región interna de color negro (0)
        #            if filled_image[y, x] == 0:  # Pixel negro (parte de fondo)
        #                cv2.floodFill(filled_image, None, (x, y), 255)  # Rellenar con blanco
        #    
        #    # Mostrar el resultado
        #    plt.figure()
        #    plt.imshow(filled_image, cmap='gray')
        #    plt.title("Imagen con Relleno de Regiones (Flood Fill)")
        #    plt.show()
#
        #    # Definir el kernel para la operación morfológica
        #    kernel = np.ones((3,3), np.uint8)  # Puedes ajustar el tamaño según los huecos
#
        #    # Aplicar operaciones de dilatación y cierre (closing)
        #    dilated = cv2.dilate(1-filled_image, kernel, iterations=1)
        #    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#
        #    # Mostrar el resultado
        #    plt.figure()
        #    plt.imshow(closed, cmap='gray')
        #    plt.title("Imagen con Operación de Cierre")
        #    plt.show()
            

        elif event.key == 'y':
            # Calcular alturas
            heights_lr = measure_heights_lr(vertebrae_extraction_normalized, pixel_spacing)
            heights_pca = measure_heights_pca(vertebrae_extraction_normalized, pixel_spacing)

            # Visualizar alturas
            plot_height_maps(heights_lr, heights_pca)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def main():
    folder_path = 'Paciente 2 - 2'  # Ajusta esta ruta según corresponda
    dicom_files = load_dicom_files(folder_path)
    if not dicom_files:
        raise ValueError(f"No se encontraron archivos DICOM en la carpeta: {folder_path}")
    dicom_files_sorted = sort_slices(dicom_files)
    img3d, img_shape, z_positions = create_3d_image(dicom_files_sorted)
    first_slice = dicom_files_sorted[0]
    if hasattr(first_slice, 'PixelSpacing'):
        pixel_spacing_y = float(first_slice.PixelSpacing[0])
        pixel_spacing_x = float(first_slice.PixelSpacing[1])
        pixel_spacing_z = float(first_slice.SliceThickness)
        pixel_spacing = (pixel_spacing_y, pixel_spacing_x, pixel_spacing_z)
        print(f"Pixel Spacing (y, x, z): {pixel_spacing_y} mm, {pixel_spacing_x} mm, {pixel_spacing_z} mm")
    else:
        raise AttributeError("Los archivos DICOM no contienen el atributo 'PixelSpacing'.")
    show_coronal_slices(img3d, img_shape, z_positions, pixel_spacing)

if __name__ == "__main__":
    main()
