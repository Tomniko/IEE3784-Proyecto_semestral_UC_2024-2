import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Cargar archivos DICOM
def load_dicom_files(folder):
    dicom_files = []
    for filename in os.listdir(folder):
        if filename.endswith('.dcm'):
            dicom_files.append(pydicom.dcmread(os.path.join(folder, filename)))
    return dicom_files

# Asegurar el orden correcto de las imágenes
def sort_slices(dicom_files):
    try:
        return sorted(dicom_files, key=lambda s: float(getattr(s, 'SliceLocation', s.ImagePositionPatient[2])))
    except AttributeError as e:
        print(f"Error al ordenar las slices: {e}")
        raise

# Crear una imagen 3D de los cortes
def create_3d_image(slices):
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
            img2d = img2d * float(s.RescaleSlope) + float(s.RescaleIntercept)
        img3d[:, :, i] = img2d
    
    return img3d, img_shape

# Obtener el corte en el plano especificado
def slice_det(image, index, plane):    
    if plane == 0: # Axial
        return image[index, :, :]
    elif plane == 1: # Sagital
        return image[:, index, :]
    elif plane == 2: # Coronal
        return image[:, :, index]

# Definición de B-Splines cúbicos
def bspline(x, t, c, k):
    n = len(t) - k - 1
    return sum(c[i] * B(x, k, i, t) for i in range(n))

def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

# Interpolación de puntos usando B-Splines cúbicos
def interpolate_spline(points, degree):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Generar nudos t
    t = np.linspace(0, len(points) - 1, len(points) + degree + 1)
    
    # Evaluar spline en un mayor número de puntos
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = np.linspace(min(y), max(y), 100)
    interp_x = [bspline(xi, t, x, degree) for xi in x_smooth]
    interp_y = [bspline(yi, t, y, degree) for yi in y_smooth]
    
    return interp_x, interp_y

# Calcular la altura de la vértebra en mm
def calculate_height(vertebra_points, pixel_spacing):
    y_coords = [pt[1] for pt in vertebra_points]
    min_y = min(y_coords)
    max_y = max(y_coords)
    delta_y = max_y - min_y
    height_mm = delta_y * pixel_spacing[0]  # pixel_spacing_y
    return height_mm

# Graficar las alturas de las vértebras
def plot_heights(heights):
    fig_heights, ax_heights = plt.subplots()
    vertebra_labels = [f'Vertebra {i+1}' for i in range(len(heights))]
    ax_heights.bar(vertebra_labels, heights, color='skyblue')
    ax_heights.set_xlabel('Vértebras')
    ax_heights.set_ylabel('Altura (mm)')
    ax_heights.set_title('Altura de los Cuerpos Vertebrales')
    for i, v in enumerate(heights):
        ax_heights.text(i, v + max(heights)*0.01, f"{v:.2f} mm", ha='center', va='bottom')
    plt.show()

# Mostrar cortes coronales con interactividad y selección de puntos
def show_coronal_slices(img3d, img_shape, pixel_spacing):
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

    slice_fixed = {'fixed': False}
    vertebrae = []
    current_vertebra = []
    markers = []
    colors = plt.cm.get_cmap('tab10', 10)
    vertebra_count = 0
    vertebra_heights = []

    def update(val):
        if not slice_fixed['fixed']:
            index = int(slider.val)
            img = slice_det(img3d, index, plane)
            img_plot.set_data(img)
            ax.set_title(f"Corte Coronal: {index}")
            fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        nonlocal vertebra_count
        if event.key == 'x' and not slice_fixed['fixed']:
            slice_fixed['fixed'] = True
            slider.ax.set_visible(False)
            fig.canvas.draw_idle()
            print(f"Corte coronal fijado en el índice: {int(slider.val)}")
            print("Ahora puedes seleccionar puntos con el mouse.")
            print("Presiona 'n' para iniciar una nueva vértebra.")
            print("Presiona 'q' para finalizar la selección de puntos.")
        elif event.key == 'n' and slice_fixed['fixed']:
            if current_vertebra:
                vertebrae.append(current_vertebra.copy())
                height_mm = calculate_height(current_vertebra, pixel_spacing)
                vertebra_heights.append(height_mm)
                current_vertebra.clear()
                vertebra_count += 1
                print(f"Iniciando selección para la vértebra {vertebra_count + 1}.")
            else:
                print("No hay puntos en la vértebra actual para guardar.")
            for idx, vertebra in enumerate(vertebrae, 1):
                if len(vertebra) < 3:
                    print(f"Vértebra {idx} tiene menos de 3 puntos. No se puede interpolar.")
                    continue
                
                x_cuad, y_cuad = interpolate_spline(vertebra, degree=2)
                x_cub, y_cub = interpolate_spline(vertebra, degree=3)
                
                if x_cuad is not None and y_cuad is not None:
                    ax.plot(x_cuad, y_cuad, label=f'Vertebra {idx} - Cuadrática', color=colors(idx % 10), linestyle='--')
                if x_cub is not None and y_cub is not None:
                    ax.plot(x_cub, y_cub, label=f'Vertebra {idx} - Cúbica', color=colors(idx % 10), linestyle='-')
                ax.legend()
                fig.canvas.draw_idle()
            print("Curvas B-Splines interpoladas y dibujadas sobre la imagen.")
        elif event.key == 'q' and slice_fixed['fixed']:
            if current_vertebra:
                vertebrae.append(current_vertebra.copy())
                height_mm = calculate_height(current_vertebra, pixel_spacing)
                vertebra_heights.append(height_mm)
                vertebra_count += 1
            print("Finalizando la selección de puntos.")
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            plot_heights(vertebra_heights)

    def on_click(event):
        if slice_fixed['fixed'] and event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            current_vertebra.append((x, y))
            color = colors(vertebra_count % 10) if vertebra_count < 10 else 'blue'
            marker = ax.scatter(x, y, c=[color])
            markers.append(marker)
            fig.canvas.draw_idle()

    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

# Cargar imágenes DICOM y mostrarlas con el slider interactivo
folder_path = 'Paciente 2 - 2'
dicom_files = load_dicom_files(folder_path)
sorted_slices = sort_slices(dicom_files)
img3d, img_shape = create_3d_image(sorted_slices)

# Suposición de espaciado de píxeles
pixel_spacing = (0.5, 0.5)  # Este valor debe leerse de los datos DICOM reales si está disponible

# Mostrar los cortes coronales con selección de puntos y B-Splines
show_coronal_slices(img3d, img_shape, pixel_spacing)
