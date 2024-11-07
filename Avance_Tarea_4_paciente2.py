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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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

def segmentar_slice_automatica(ct_slice, initial_threshold, max_iter=10, window_size=2):
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


def segmentacion_mumford_shah(slice_img, iteraciones=200, lambda1=1, lambda2=1):
    # Preprocesamiento: Suavizar la imagen para reducir el ruido
    img_suavizada = cv2.GaussianBlur(slice_img, (5,5), 0.5)

    # Inicializar el nivel de conjunto inicial (checkerboard)
    init_level_set = checkerboard_level_set(slice_img.shape, 4  )

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


# Función para medir la altura en la dirección pies-cabeza (eje vertical)
def measure_height_lr(y_coords, pixel_spacing):
    return (y_coords.max() - y_coords.min()) * pixel_spacing[0]

# Función para medir la altura en la dirección principal (PCA)
def measure_height_pca(x_coords, y_coords, pixel_spacing):
    points = np.column_stack((x_coords, y_coords))
    pca = PCA(n_components=2)
    pca.fit(points)
    projections = pca.transform(points)
    return (projections[:, 0].max() - projections[:, 0].min()) * pixel_spacing[0]

## Función para obtener alturas y graficar en escala de grises o código de color
def calculate_and_plot_heights(mark, pixel_spacing):
    unique_regions = np.unique(mark)
    unique_regions = unique_regions[unique_regions != 0]  # Ignorar el fondo
    
    # Crear una máscara para almacenar las alturas
    height_lr_map = np.zeros_like(mark, dtype=np.float32)
    height_pca_map = np.zeros_like(mark, dtype=np.float32)
    
    # Calcular alturas y almacenar en las máscaras correspondientes
    for region_label in unique_regions:
        # Crear máscara para el segmento actual
        region_mask = (mark == region_label)
        y_coords, x_coords = np.where(region_mask)
        
        # Medir alturas
        height_lr = measure_height_lr(y_coords, pixel_spacing)
        height_pca = measure_height_pca(x_coords, y_coords, pixel_spacing)
        
        # Asignar valores de altura en cada píxel de la vértebra actual en las máscaras
        height_lr_map[region_mask] = height_lr
        height_pca_map[region_mask] = height_pca

    # Configurar los valores para visualización
    norm_lr = Normalize(vmin=height_lr_map.min(), vmax=height_lr_map.max())
    norm_pca = Normalize(vmin=height_pca_map.min(), vmax=height_pca_map.max())
    cmap = plt.cm.gray  # Cambiar a plt.cm.jet o cualquier mapa de color deseado

    # Visualizar altura pies-cabeza y añadir etiquetas de altura
    fig_lr, ax_lr = plt.subplots()
    ax_lr.set_title("Altura en Dirección Pies-Cabeza")
    img_lr = ax_lr.imshow(height_lr_map, cmap=cmap, norm=norm_lr)
    fig_lr.colorbar(ScalarMappable(norm=norm_lr, cmap=cmap), ax=ax_lr, label='Altura (mm)')
    ax_lr.axis('off')
    
    # Añadir etiquetas de altura en el centro de cada segmento
    for region_label in unique_regions:
        region_mask = (mark == region_label)
        y_coords, x_coords = np.where(region_mask)
        height_lr = measure_height_lr(y_coords, pixel_spacing)
        # Calcular el centro del segmento para colocar la etiqueta
        y_center, x_center = np.mean(y_coords).astype(int), np.mean(x_coords).astype(int)
        ax_lr.text(x_center, y_center, f"{height_lr:.1f} mm", color="red", ha="center", va="center", fontsize=8)

    # Visualizar altura en dirección principal (PCA) y añadir etiquetas de altura
    fig_pca, ax_pca = plt.subplots()
    ax_pca.set_title("Altura en Dirección Principal (PCA)")
    img_pca = ax_pca.imshow(height_pca_map, cmap=cmap, norm=norm_pca)
    fig_pca.colorbar(ScalarMappable(norm=norm_pca, cmap=cmap), ax=ax_pca, label='Altura (mm)')
    ax_pca.axis('off')
    
    # Añadir etiquetas de altura en el centro de cada segmento para la dirección PCA
    for region_label in unique_regions:
        region_mask = (mark == region_label)
        y_coords, x_coords = np.where(region_mask)
        height_pca = measure_height_pca(x_coords, y_coords, pixel_spacing)
        # Calcular el centro del segmento para colocar la etiqueta
        y_center, x_center = np.mean(y_coords).astype(int), np.mean(x_coords).astype(int)
        ax_pca.text(x_center, y_center, f"{height_pca:.1f} mm", color="red", ha="center", va="center", fontsize=8)

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
    mark = None

    def on_key(event):
        nonlocal slice_img, mask, selected_slice, vertebrae_extraction_normalized, closed, mark
        if event.key == 'x':
            selected_slice = int(slider.val)
            print(f"Corte coronal fijado en el índice: {selected_slice}")
        elif event.key == 'r':
            selected_slice = int(slider.val)
            slice_img = slice_det(img3d, selected_slice, plane)
            img_suavizada = cv2.GaussianBlur(slice_img, (1,1), 0.5)
            mask = segmentar_slice_automatica(img_suavizada, initial_threshold=112)
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
            # Aplicar operaciones de dilatación y cierre (closing)
            dilated = cv2.dilate(filled_image, kernel, iterations=1)
            closed = cv2.erode(dilated, kernel, iterations=1)
            # Mostrar el resultado
            plt.figure()
            plt.imshow(closed, cmap='gray')
            plt.title("Imagen con Operación de Cierre")
            plt.show()

            #mostrar_histograma(closed)
            
            # Calculamos el histograma para la máscara closed
            hist, bins = np.histogram(closed, bins=[0, 1, 2])

            # Obtenemos la cantidad de valores 0 y 1
            count_zeros = hist[0]
            count_ones = hist[1]

            # Aplicamos la condicionalidad
            if count_ones < count_zeros:
                vertebrae_extraction = slice_img * closed
            else:
                vertebrae_extraction = slice_img * (1 - closed)
            vertebrae_extraction_normalized = cv2.normalize(vertebrae_extraction, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #threshold 100 y 110 en sagital
            #threshold 140 y 150 en coronal
            # MEJOR DE TODAS ES LA SLICE 341 CORONAL
            # MEJOR SLICE 269 SAGITAL
            plt.figure()
            plt.imshow(vertebrae_extraction_normalized, cmap='gray')
            plt.title(f"Vértebras Extraída con Contorno - Slice {selected_slice}")
            plt.show()

            binary_image_segmentation_mask = 1 - closed
            height, width = binary_image_segmentation_mask.shape
            ### ROI PARA CORONAL
            ##roi = vertebrae_extraction_normalized_tst[int(height * 0.3):int(height * 0.7), :int(width * 0.9)]

            ### ROI PARA SAGITAL
            roi = binary_image_segmentation_mask[int(height * 0.5):int(height * 0.9), :int(width * 0.9)]
            # Aplicar la transformada de distancia para identificar el centro de las regiones
            dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
            # Normalizar la imagen de distancia para que esté en el rango 
            cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
            plt.figure()
            plt.title("Imagen de Distancia Normalizada")
            plt.imshow(dist, cmap='gray')
            plt.show()
            mostrar_histograma(dist, "Histograma de la Imagen de Distancia")
            # Aplicar un umbral para obtener los picos, que serán los marcadores para los objetos de primer plano
            _, dist_bin  = cv2.threshold(dist, 200, 255, cv2.THRESH_BINARY)
            # Convertir la imagen de distancia a tipo uint8 para encontrar contornos
            dist_8u = dist_bin.astype(np.uint8)
            # Encontrar contornos en la imagen umbralizada
            contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Crear la imagen de marcadores para el algoritmo watershed
            markers = np.zeros(dist.shape, dtype=np.int32)
            # Dibujar los marcadores de los objetos de primer plano
            for i in range(len(contours)):
                cv2.drawContours(markers, contours, i, i + 1, -1)  # Los índices empiezan en 1
            # Dibujar un marcador para el fondo (parte superior izquierda de la imagen)
            cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
            # Visualizar los marcadores antes de aplicar watershed (opcional)
            markers_8u = (markers * 10).astype('uint8')
            plt.figure()
            plt.title("Markers")
            plt.imshow(markers_8u, cmap='gray')
            plt.show()
            # Aplicar el algoritmo de watershed
            imgResult = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convertir la imagen a color para watershed
            cv2.watershed(imgResult, markers)
            # Convertir los marcadores a uint8 y hacer una inversión para visualizar (opcional)
            mark = markers.astype('uint8')
            mark = cv2.bitwise_not(mark)
            
            
            # Generar colores aleatorios para cada región detectada
            random.seed(42)
            colors = []
            for contour in contours:
                colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            # Crear la imagen de resultado
            dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
            # Rellenar los objetos etiquetados con colores aleatorios
            for i in range(markers.shape[0]):
                for j in range(markers.shape[1]):
                    index = markers[i, j]
                    if index > 0 and index <= len(contours):
                        dst[i, j, :] = colors[index - 1]
            # Mostrar la imagen final con los objetos coloreados
            plt.figure()
            plt.title("Final Result")
            plt.imshow(dst, cmap='gray')
            plt.show()
            # Dilatar la imagen de distancia para resaltar los picos (opcional)
            kernel1 = np.ones((3, 3), np.uint8)
            dist = cv2.dilate(dist, kernel1)
            closed = cv2.erode(dist, kernel1, iterations=2)
            plt.figure()
            plt.title("Peaks")
            plt.imshow(closed, cmap='gray')
            plt.show()
             

        elif event.key == 'y':
            calculate_and_plot_heights(mark,pixel_spacing)



    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def main():
    folder_path = 'Paciente 2 - 2'  # Ajusta esta ruta según corresponda
    #PACIENTE 1 274 BUENO
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
