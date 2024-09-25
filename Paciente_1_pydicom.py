import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
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

# Definir las funciones base B-spline utilizando la recursión descrita en el documento
def B_spline(i, k, t, x):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    else:
        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]
        term1 = 0 if denom1 == 0 else (x - t[i]) / denom1 * B_spline(i, k-1, t, x)
        term2 = 0 if denom2 == 0 else (t[i+k+1] - x) / denom2 * B_spline(i+1, k-1, t, x)
        return term1 + term2

# Interpolación de B-splines cúbicos periódicos
def periodic_bspline_interpolation(points, degree):
    points = np.vstack([points[-degree:], points, points[:degree]])
    n = len(points)
    
    k = degree
    t = np.linspace(0, n, n + k + 1)

    x_vals = np.linspace(0, n, 100)
    spline_x = np.zeros_like(x_vals)
    spline_y = np.zeros_like(x_vals)

    for i in range(n):
        spline_x += points[i, 0] * np.array([B_spline(i, k, t, x) for x in x_vals])
        spline_y += points[i, 1] * np.array([B_spline(i, k, t, x) for x in x_vals])
    
    if degree == 2:
        return spline_x[degree*8:-degree*8], spline_y[degree*8:-degree*8]
    if degree == 3:
        return spline_x[degree*6:-degree*6], spline_y[degree*6:-degree*6]

# Crear máscara desde la spline
def create_mask_from_spline(spline_x, spline_y, img_shape):
    spline_path = Path(np.column_stack((spline_x, spline_y)))
    x, y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = spline_path.contains_points(points)
    mask = mask.reshape(img_shape)
    return mask

def plot_heights(heights):
    fig_heights, ax_heights = plt.subplots()
    values_x = [i for i in range(len(heights))]
    ax_heights.plot(values_x, heights, marker='o', color='skyblue', linestyle='-', linewidth=2)
    ax_heights.set_xlabel('Vértebras')
    ax_heights.set_ylabel('Altura (mm)')
    ax_heights.set_title('Altura de los Cuerpos Vertebrales')
    for i, v in enumerate(heights):
        ax_heights.text(i, v + max(heights)*0.01, f"{v:.2f} mm", ha='center', va='bottom')
    plt.show()

# Mostrar cortes coronales con interactividad
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
    masked_image = None  # Variable para la imagen recortada
    colors = plt.cm.get_cmap('tab10', 10)
    vertebra_count = 0

    def update(val):
        if not slice_fixed['fixed']:
            index = int(slider.val)
            img = slice_det(img3d, index, plane)
            img_plot.set_data(img)
            ax.set_title(f"Corte Coronal: {index}")
            fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        nonlocal vertebra_count, masked_image
        if event.key == 'x' and not slice_fixed['fixed']:
            slice_fixed['fixed'] = True
            slider.ax.set_visible(False)
            fig.canvas.draw_idle()
            print(f"Corte coronal fijado en el índice: {int(slider.val)}")
        elif event.key == 'n' and slice_fixed['fixed']:
            if current_vertebra:
                vertebrae.append(current_vertebra.copy())
                current_vertebra.clear()
                vertebra_count += 1
                print(f"Iniciando selección para la vértebra {vertebra_count + 1}.")
        elif event.key == 'r' and slice_fixed['fixed']:
            try:
                # Solicitar al usuario el grado de la B-spline
                degree = int(input("Introduce el grado de la B-spline (ejemplo: 2, 3): "))
                
                for idx, vertebra in enumerate(vertebrae, 1):
                    if len(vertebra) < 3:
                        print(f"Vértebra {idx} tiene menos de 3 puntos. No se puede interpolar.")
                        continue
                        
                    # Usar el grado especificado por el usuario para la interpolación
                    x_cub, y_cub = periodic_bspline_interpolation(vertebra, degree=degree)
        
                    if x_cub is not None and y_cub is not None:
                        ax.plot(x_cub, y_cub, label=f'Vertebra {idx} - Spline (grado {degree})', color=colors(idx % 10), linestyle='-')
                ax.legend()
                fig.canvas.draw_idle()
                print(f"Curvas B-Splines (grado {degree}) dibujadas sobre la imagen.")
        
            except ValueError:
                print("Por favor, introduce un número entero válido para el grado.")

        elif event.key == 'c' and slice_fixed['fixed']:
            if vertebrae:
                # Usar la última vértebra añadida
                last_vertebra = vertebrae[-1]
                # Solicitar al usuario el grado de la B-spline
                degree = int(input("Introduce el grado de la B-spline para la máscara (ejemplo: 2, 3): "))
                x_cub, y_cub = periodic_bspline_interpolation(last_vertebra, degree=degree)
                mask = create_mask_from_spline(x_cub, y_cub, img_display.shape)

                # Aplicar la máscara a la imagen
                masked_image = np.copy(img_display)
                masked_image[~mask] = 0
                ax.imshow(masked_image, cmap='gray')
                fig.canvas.draw_idle()
                print("Región encerrada por la B-spline recortada y mostrada.")
        elif event.key == 'q' and slice_fixed['fixed']:
            print("Finalizando la selección de puntos.")
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)

            if masked_image is not None:
                # Crear una lista para almacenar las alturas proyectadas en cada x
                projected_heights = []

                
                # Recorrer todas las coordenadas x dentro de la región de la máscara
                for x in range(masked_image.shape[1]):  # Recorrer cada columna (x)
                    y_coords = np.where(masked_image[:, x] > 0)[0]  # Obtener los valores de y en donde la imagen tiene intensidades (> 0)
                    if len(y_coords) > 0:  # Si hay valores no nulos en esta columna
                        min_y = np.min(y_coords)
                        max_y = np.max(y_coords)
                        height_mm = (max_y - min_y) * pixel_spacing[0]  # Calcular la altura en mm
                        projected_heights.append((x, height_mm))  # Guardar la altura asociada a este x
                    else:
                        projected_heights.append((x, 0))  # Si no hay altura, se almacena 0

                # Separar las coordenadas x y las alturas para graficarlas
                x_vals, heights = zip(*projected_heights)

                # Graficar la proyección de las alturas en función de x
                fig2, ax2 = plt.subplots()
                ax2.plot(x_vals, heights, label="Proyección de la altura en cada x", color="blue")
                ax2.set_xlabel("Coordenada X (píxeles)")
                ax2.set_ylabel("Altura (mm)")
                ax2.set_title("Proyección de la Altura en función de X")
                ax2.legend()
                # Ajustar los límites de los ejes
                # Encontrar el primer y último valor x donde y_coords es mayor a 0
                non_zero_x = [x for x, h in zip(x_vals, heights) if h > 0]

                # Si hay valores no nulos en x, ajustar el zoom, de lo contrario usar el rango completo
                if len(non_zero_x) > 0:
                    x_min_zoom = max(min(non_zero_x) - 50, 0)  # Zoom mínimo: hasta 50 unidades antes del primer valor
                    x_max_zoom = min(max(non_zero_x) + 50, masked_image.shape[1])  # Zoom máximo: hasta 50 unidades después del último valor
                else:
                    x_min_zoom = 0
                    x_max_zoom = masked_image.shape[1]  # Usar el rango completo si no hay valores no nulos

                # Ajustar los límites del eje X para hacer zoom
                ax2.set_xlim([x_min_zoom, x_max_zoom])
                fig2.canvas.draw() 
                plt.show()

    def on_click(event):
        if slice_fixed['fixed'] and event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            current_vertebra.append((x, y))
            color = colors(vertebra_count % 10) if vertebra_count < 10 else 'black'
            marker, = ax.plot(x, y, marker='o', markersize=5, markerfacecolor=color, markeredgecolor='white')
            markers.append(marker)
            fig.canvas.draw_idle()
            print(f"Punto seleccionado en (x={x}, y={y}) para la vértebra {vertebra_count + 1}.")


    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    folder_path = 'Paciente 2 - 2'
    dicom_files = load_dicom_files(folder_path)
    if not dicom_files:
        raise ValueError(f"No se encontraron archivos DICOM en la carpeta: {folder_path}")
    dicom_files_sorted = sort_slices(dicom_files)
    img3d, img_shape = create_3d_image(dicom_files_sorted)
    first_slice = dicom_files_sorted[0]
    if hasattr(first_slice, 'PixelSpacing'):
        pixel_spacing_y = float(first_slice.PixelSpacing[0])
        pixel_spacing_x = float(first_slice.PixelSpacing[1])
        pixel_spacing = (pixel_spacing_y, pixel_spacing_x)
        print(f"Pixel Spacing (y, x): {pixel_spacing_y} mm, {pixel_spacing_x} mm")
    else:
        raise AttributeError("Los archivos DICOM no contienen el atributo 'PixelSpacing'.")
    show_coronal_slices(img3d, img_shape, pixel_spacing)

if __name__ == "__main__":
    main()
