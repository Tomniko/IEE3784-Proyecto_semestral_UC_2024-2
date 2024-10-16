import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

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

# Función para generar la superficie B-spline utilizando interpolación cúbica
def generate_b_spline_surface(points_3d, resolution=100):
    """
    Genera una superficie B-spline a partir de puntos 3D utilizando interpolación cúbica.
    
    Args:
        points_3d (ndarray): Puntos 3D de la vértebra (N, 3).
        resolution (int): Resolución de la grilla.
    
    Returns:
        tuple: (grid_x, grid_y, grid_z)
    """
    # Separar las coordenadas x, y y z de los puntos 3D
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    # Crear una grilla para los puntos xy
    grid_x, grid_y = np.mgrid[min(x):max(x):complex(resolution), min(y):max(y):complex(resolution)]
    
    # Interpolación B-Spline para obtener valores de z en la grilla
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    # Reemplazar NaN con interpolación 'nearest'
    mask_nan = np.isnan(grid_z)
    if np.any(mask_nan):
        grid_z[mask_nan] = griddata((x, y), z, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')
    
    return grid_x, grid_y, grid_z

# Crear máscara desde la spline
def create_mask_from_spline(spline_x, spline_y, img_shape):
    spline_path = Path(np.column_stack((spline_x, spline_y)))
    x, y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = spline_path.contains_points(points)
    mask = mask.reshape(img_shape)
    return mask

# Función para medir altura considerando la dirección principal de la vértebra
def measure_heights_principal(surface_x, surface_y, surface_z, pixel_spacing):
    """
    Calcula la altura de la vértebra considerando la dirección principal usando PCA.
    
    Args:
        surface_x (ndarray): Coordenadas X de la superficie B-spline.
        surface_y (ndarray): Coordenadas Y de la superficie B-spline.
        surface_z (ndarray): Coordenadas Z de la superficie B-spline.
        pixel_spacing (tuple): Espaciado de píxeles en (y, x, z) en mm.
    
    Returns:
        tuple: (height_image_mm, height_mm)
            - height_image_mm (ndarray): Imagen 2D de alturas en mm.
            - height_mm (float): Altura total en mm.
    """
    # Apilar las coordenadas en una matriz (N, 3)
    points = np.column_stack((surface_x.ravel(), surface_y.ravel(), surface_z.ravel()))
    
    # Crear máscara para puntos válidos (no NaN ni Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    valid_points = points[valid_mask]
    
    if valid_points.shape[0] < 2:
        print("No hay suficientes puntos válidos para realizar PCA.")
        return None, None
    
    # PCA para encontrar la dirección principal en el plano XY
    pca = PCA(n_components=2)
    pca.fit(valid_points[:, :2])  # PCA en el plano XY
    
    principal_axis = pca.components_[0]
    secondary_axis = pca.components_[1]
    
    # Crear matriz de rotación para alinear el eje principal con el eje X
    rotation_matrix = np.array([
        [principal_axis[0], principal_axis[1], 0],
        [secondary_axis[0], secondary_axis[1], 0],
        [0, 0, 1]
    ])
    
    # Rotar todos los puntos de la superficie
    grid_points = np.column_stack((surface_x.ravel(), surface_y.ravel(), surface_z.ravel()))
    grid_points_rot = grid_points @ rotation_matrix
    
    # Reshape de las coordenadas rotadas
    try:
        grid_z_rot = grid_points_rot[:, 2].reshape(surface_x.shape)
    except ValueError as e:
        print(f"Error al remodelar grid_z_rot: {e}")
        return None, None
    
    # Calcular la altura como la diferencia entre el máximo y mínimo en Z rotado
    height = np.nanmax(grid_z_rot) - np.nanmin(grid_z_rot)
    
    # Convertir la altura a milímetros
    height_mm = height * pixel_spacing[2]
    
    # Crear una imagen de alturas por píxel (z_rot - z_min)
    height_image = grid_z_rot - np.nanmin(grid_z_rot)
    height_image_mm = height_image * pixel_spacing[2]
    
    return height_image_mm, height_mm

# Visualización de las alturas en color o escala de grises
def plot_height_images_principal(height_image_mm, pixel_spacing):
    """
    Visualiza la imagen de alturas utilizando un mapa de colores.
    
    Args:
        height_image_mm (ndarray): Imagen 2D de alturas en mm.
        pixel_spacing (tuple): Espaciado de píxeles en (y, x, z) en mm.
    """
    if height_image_mm is None:
        print("No se pudo generar la imagen de alturas.")
        return
    
    # Reemplazar NaN y Inf con 0
    height_image_mm = np.nan_to_num(height_image_mm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Obtener dimensiones
    height, width = height_image_mm.shape
    
    # Crear ejes físicos
    extent = [
        0, width * pixel_spacing[1],  # X-axis
        0, height * pixel_spacing[0]  # Y-axis
    ]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(height_image_mm, cmap='Grays', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im, label='Altura (mm)')
    plt.title("Altura en Dirección Principal")
    plt.xlabel("Distancia Principal (mm)")
    plt.ylabel("Distancia Secundaria (mm)")
    plt.tight_layout()
    plt.show()

# Graficar la superficie B-spline
def plot_3d_surface(grid_x, grid_y, grid_z):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='gray', edgecolor='none')
    ax.set_xlabel('X (Izquierda-Derecha) [mm]')
    ax.set_ylabel('Y (Anteroposterior) [mm]')
    ax.set_zlabel('Z (Altura) [mm]')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Altura (mm)')
    plt.show()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_height_images_lr_ap(grid_x, grid_y, grid_z, pixel_spacing):
    # Convertir las coordenadas a unidades físicas (mm)
    grid_x_mm = grid_x * pixel_spacing[0]
    grid_y_mm = grid_y * pixel_spacing[1]
    grid_z_mm = grid_z * pixel_spacing[2]

    plt.figure(figsize=(12, 6))

    # Superficie Izquierda-Derecha (X vs Z), mostrando la variación en Y
    plt.subplot(1, 2, 1)
    plt.contourf(grid_x_mm, grid_z_mm, grid_y_mm, cmap='gray')
    plt.colorbar(label='Distancia Anteroposterior (Y) (mm)')
    plt.title("Superficie Izquierda-Derecha (X vs Z)")
    plt.xlabel("Distancia Izquierda-Derecha (mm)")
    plt.ylabel("Altura (Z) (mm)")

    # Superficie Anteroposterior (X vs Y), mostrando la variación en Z
    plt.subplot(1, 2, 2)
    plt.contourf(grid_x_mm, grid_y_mm, grid_z_mm, cmap='gray')
    plt.colorbar(label='Altura (Z) (mm)')
    plt.title("Superficie Anteroposterior (X vs Y)")
    plt.xlabel("Distancia Izquierda-Derecha (mm)")
    plt.ylabel("Distancia Anteroposterior (Y) (mm)")

    plt.tight_layout()
    plt.show()


    
# Mostrar cortes coronales
def show_coronal_slices(img3d, img_shape, z_positions, pixel_spacing):
    plane = 0  # Coronal
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

    def on_key(event):
        nonlocal vertebra_count, current_vertebra, bspline_surfaces
        if event.key == 'x':
            print(f"Corte coronal fijado en el índice: {int(slider.val)}")
        elif event.key == 'n':
            if current_vertebra:
                vertebrae.append(current_vertebra.copy())
                current_vertebra.clear()
                vertebra_count += 1
                print(f"Vértebra {vertebra_count} añadida.")
            else:
                vertebra_count += 1
                vertebrae.append({})
                print(f"Vértebra {vertebra_count} iniciada.")
        elif event.key == 't': 
            list_points = input("¿Utilizar lista de puntos predeterminados? (s/n): ")
            if list_points.lower() == 's':
                default_file_path = 'Ptos_prueba_predeterminados.txt'
                try:
                    with open(default_file_path, 'r') as file:
                        points = [line.strip() for line in file.readlines()]
                        formatted_points = []
                        for point in points:
                            x, y, ref = map(int, point.strip('()').split(','))
                            formatted_points.append((y, x, ref))

                        current_vertebra = {}
                        for point in formatted_points:
                            ref = point[2]
                            if ref not in current_vertebra:
                                current_vertebra[ref] = []
                            current_vertebra[ref].append((point[0], point[1]))
                    
                        vertebrae.append(current_vertebra)
                        vertebra_count += 1
                        print(f"Se ha cargado la vértebra {vertebra_count} con puntos predeterminados.")
                        #print(vertebrae)
                except FileNotFoundError:
                    print("Archivo no encontrado. Por favor, asegúrate de que el archivo 'Ptos_prueba_predeterminados.txt' existe en el directorio.")
            else:
                current_vertebra = {}
                vertebrae.append(current_vertebra)
                vertebra_count += 1
                print(f"Iniciando selección grupal para la vértebra {vertebra_count}.")
        elif event.key == 'r':
            list_points = input("¿Graficar con puntos predeterminados? (s/n): ")
            if vertebrae:
                for idx, vertebra in enumerate(vertebrae, 1):
                    points_3d = []
                    for slice_idx, points in vertebra.items():
                        if slice_idx >= len(z_positions):
                            print(f"Slice index {slice_idx} fuera de rango. Skipping.")
                            continue
                        
                        z = slice_idx
                        for (x, y) in points:
                            # Convertir a unidades físicas
                            x_phys = x
                            y_phys = y
                            points_3d.append([x_phys, y_phys, z])
                    points_3d = np.array(points_3d)
                    if len(points_3d) < 4:
                        print(f"No hay suficientes puntos para generar una superficie B-spline para la vértebra {idx}. Se requieren al menos 4 puntos.")
                        continue
                    # Generar la superficie B-spline utilizando la función proporcionada
                    try:
                        grid_x, grid_y, grid_z = generate_b_spline_surface(points_3d)
                        bspline_surfaces.append((grid_x, grid_y, grid_z))
                        print(f"Superficie B-spline para la vértebra {idx} generada.")
                        #print(bspline_surfaces)
                    except Exception as e:
                        print(f"Error al generar la superficie B-spline para la vértebra {idx}: {e}")
        elif event.key == 'y':
            if bspline_surfaces:
                for grid_x, grid_y, grid_z in bspline_surfaces:
                    plot_3d_surface(grid_x, grid_y, grid_z)
                print("Superficies B-spline generadas y mostradas en 3D.")
            else:
                print("No se generaron superficies B-spline.")
        elif event.key == 'q':
            if bspline_surfaces:
                for idx, (grid_x, grid_y, grid_z) in enumerate(bspline_surfaces, 1):
                    # Medir altura considerando la dirección principal
                    height_image_mm, height_mm = measure_heights_principal(grid_x, grid_y, grid_z, pixel_spacing)
                    if height_image_mm is not None:
                        print(f"Altura de la vértebra {idx}: {height_mm:.2f} mm")
                        plot_height_images_lr_ap(grid_x, grid_y, grid_z, pixel_spacing)
                        # Visualizar las alturas
                        plot_height_images_principal(height_image_mm, pixel_spacing)
                    else:
                        print(f"No se pudo medir la altura de la vértebra {idx}.")
            else:
                print("No se generaron superficies B-spline.")
                        
                    
    def on_click(event):
        if event.inaxes == ax:
            slice_idx = int(slider.val)
            try:
                x = int(event.xdata)
                y = int(event.ydata)
            except (ValueError, TypeError):
                print("Clic fuera del rango de la imagen.")
                return
            if vertebra_count == 0:
                print("No hay vértebras creadas. Presiona 'n' para crear una nueva vértebra.")
                return
            last_vertebra = vertebrae[-1]
            if slice_idx not in last_vertebra:
                last_vertebra[slice_idx] = []
            last_vertebra[slice_idx].append((x, y))
            color = colors_cmap(vertebra_count % 10) if vertebra_count < 10 else 'black'
            if slice_idx not in markers:
                markers[slice_idx] = []
            marker, = ax.plot(x, y, marker='o', markersize=5, markerfacecolor=color, markeredgecolor='white')
            markers[slice_idx].append(marker)
            fig.canvas.draw_idle()
            print(f"Punto seleccionado en (x={x}, y={y}, z={slice_idx}) para la vértebra {vertebra_count}.")

    fig.canvas.mpl_connect('button_press_event', on_click)
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
