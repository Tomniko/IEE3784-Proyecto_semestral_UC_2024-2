# Segmentación Automática y Medición de Altura Vertebral en Imágenes de Tomografía Computarizada usando Métodos Adaptativos y PCA
Este código permite cargar, procesar, y analizar imágenes de cortes tomográficos en formato DICOM, enfocándose en la segmentación y medición de alturas vertebrales. Incluye funciones para cargar y ordenar los archivos DICOM, crear imágenes tridimensionales a partir de los cortes, y realizar segmentación en 2D de estructuras óseas mediante métodos adaptativos y de Mumford-Shah. Además, se integran herramientas de visualización interactivas para examinar los cortes y analizar las alturas vertebrales en diferentes direcciones.

## Requisitos
Para ejecutar este código, se deben instalar las siguientes librerías:
```bash 
pip install pydicom
pip install numpy
pip install scipy
pip install scikit-image
pip install matplotlib
pip install opencv-python
pip install sklearn
```

## Descripción de las Funciones Principales
--- load_dicom_files(folder): Carga archivos DICOM desde la carpeta especificada. Filtra y añade solo los archivos válidos en una lista de diccionarios.

sort_slices(dicom_files): Ordena los cortes según la posición de la imagen en el paciente, garantizando la coherencia en el orden de los cortes.

create_3d_image(slices): Genera una imagen tridimensional y un vector de posiciones Z a partir de una lista de cortes DICOM.

slice_det(image, index, plane): Extrae y devuelve un corte 2D específico (coronal, sagital, o axial) de la imagen tridimensional.

mostrar_histograma(imagen, titulo): Muestra un histograma de las intensidades de un corte dado, útil para análisis preliminares de intensidad.

segmentar_slice_automatica(ct_slice, initial_threshold, max_iter, window_size): Realiza la segmentación de huesos en un corte de imagen de CT usando umbralización adaptativa.

segmentacion_mumford_shah(slice_img, iteraciones, lambda1, lambda2): Segmenta un corte de imagen utilizando una variación del método de Mumford-Shah (aproximado por el método de Chan-Vese).

measure_height_lr(y_coords, pixel_spacing) y measure_height_pca(x_coords, y_coords, pixel_spacing): Calculan la altura de una región segmentada en dos direcciones:

measure_height_lr: A lo largo del eje pies-cabeza.
measure_height_pca: En la dirección principal calculada con PCA.
calculate_and_plot_heights(mark, pixel_spacing): Calcula las alturas de cada segmento vertebral identificado en la dirección pies-cabeza y PCA. Muestra estos valores en un mapa de color.

show_coronal_slices(img3d, img_shape, z_positions, pixel_spacing): Visualiza cortes coronales con un slider interactivo. Incluye una función para activar la segmentación automática y la segmentación de Mumford-Shah con pulsaciones de teclado (x, r, b).

## Uso
Para ejecutar el código, siga los pasos:

1. Cargue los archivos DICOM con load_dicom_files("ruta/carpeta").
2. Ordene los cortes usando sort_slices(dicom_files).
3. Genere la imagen 3D y extraiga posiciones Z con create_3d_image(slices).
4. Visualice cortes específicos o aplique segmentación con las funciones de segmentación y análisis de altura.
5. Controles Interactivos
- Slider: Permite navegar por los cortes coronales.
- Teclas:
  - <kbd>r</kbd>: Realiza segmentación automática en el corte actual.
  - <kbd>b</kbd>: Aplica segmentación de Mumford-Shah al corte segmentado.
  - <kbd>y</kbd>: Realiza los gráficos de mediciones de alturas de pies a cabeza y con PCA
