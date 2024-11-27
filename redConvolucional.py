import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuración
IMG_DIR = "./Mastocitos_size"  # Directorio de las imágenes originales
MASK_DIR = "./ImagenesBinarias"  # Directorio de las máscaras binarias
IMG_SIZE = 512  # Tamaño de las imágenes (512x512)
BATCH_SIZE = 16
EPOCHS = 20

# Función para cargar imágenes y máscaras
def cargar_datos(img_dir, mask_dir, img_size=(IMG_SIZE, IMG_SIZE)):
    imagenes = []
    mascaras = []

    for archivo in os.listdir(img_dir):
        # Cargar la imagen original
        img_path = os.path.join(img_dir, archivo)
        img = load_img(img_path, target_size=img_size)  # Cargar y redimensionar si es necesario
        img = img_to_array(img) / 255.0  # Normalizar a [0, 1]
        imagenes.append(img)

        # Cargar la máscara correspondiente
        mask_name = archivo.replace('.jpg', '.png')  # Asegurar nombres coincidentes
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):  # Asegurar que la máscara exista
            mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
            mask = img_to_array(mask) / 255.0  # Normalizar a [0, 1]
            mascaras.append(mask)
        else:
            print(f"Advertencia: Máscara no encontrada para {archivo}")
            continue

    return np.array(imagenes), np.array(mascaras)

# Cargar imágenes y máscaras
imagenes, mascaras = cargar_datos(IMG_DIR, MASK_DIR)

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(imagenes, mascaras, test_size=0.2, random_state=42)

# Confirmar tamaños
print(f"Imágenes de entrenamiento: {len(X_train)}, Imágenes de validación: {len(X_val)}")

# Función para definir el modelo U-Net
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = tf.keras.layers.Input(input_size)

    # Bloque de codificación
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Capa intermedia
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Bloque de decodificación
    u5 = tf.keras.layers.UpSampling2D((2, 2))(c4)
    u5 = tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same')(u5)
    u5 = tf.keras.layers.Concatenate()([u5, c3])

    u6 = tf.keras.layers.UpSampling2D((2, 2))(u5)
    u6 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same')(u6)
    u6 = tf.keras.layers.Concatenate()([u6, c2])

    u7 = tf.keras.layers.UpSampling2D((2, 2))(u6)
    u7 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same')(u7)
    u7 = tf.keras.layers.Concatenate()([u7, c1])

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u7)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Instanciar el modelo
model = unet_model()

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Guardar el modelo entrenado
model.save("modelo_unet_mastocitos_512.h5")

# Visualizar resultados
def mostrar_resultados(imagenes, mascaras, predicciones, num=3):
    for i in range(num):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Imagen Original")
        plt.imshow(imagenes[i])
        
        plt.subplot(1, 3, 2)
        plt.title("Máscara Real")
        plt.imshow(mascaras[i].squeeze(), cmap="gray")
        
        plt.subplot(1, 3, 3)
        plt.title("Predicción")
        plt.imshow(predicciones[i].squeeze(), cmap="gray")
        
        plt.show()

# Predicción en imágenes de validación
predicciones = model.predict(X_val)
mostrar_resultados(X_val, y_val, predicciones)