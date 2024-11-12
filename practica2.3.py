import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


def momentos(image):
    image_gray = image.convert("L")
    data = list(image_gray.getdata())
    n = len(data)

    suma = sum(data)
    media = suma / n

    suma_cuadrados = sum((x - media) ** 2 for x in data)
    varianza = suma_cuadrados / n

    suma_asimetria = sum((x - media) ** 3 for x in data)
    asimetria = (suma_asimetria / n) / (varianza ** 1.5) if varianza != 0 else 0

    suma_curtosis = sum((x - media) ** 4 for x in data)
    curtosis = (suma_curtosis / n) / (varianza ** 2) - 3 if varianza != 0 else -3

    return media, varianza, asimetria, curtosis


def descriptores(image, distancia=1, angulo=0):
    image_gray = image.convert("L")
    data = np.array(image_gray)
    
    glcm = graycomatrix(data, distances=[distancia], angles=[np.deg2rad(angulo)], levels=256, symmetric=True, normed=True)

    energia = graycoprops(glcm, 'energy')[0, 0]
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]

    glcm_normalizada = glcm / glcm.sum()
    entropia = -np.sum(glcm_normalizada * np.log2(glcm_normalizada + np.finfo(float).eps))

    return energia, contraste, entropia, homogeneidad


def procesar_conjunto(directorio):
    resultados = []
    for archivo in os.listdir(directorio):
        if archivo.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            imagen = Image.open(os.path.join(directorio, archivo))
            media, varianza, asimetria, curtosis = momentos(imagen)
            energia, contraste, entropia, homogeneidad = descriptores(imagen)
            resultados.append([archivo, media, varianza, asimetria, curtosis, energia, contraste, entropia, homogeneidad])
    return resultados

directorio_conjunto1 = r"C:\Users\Anna Beristain\Downloads\practica2.3Vision\conjunto1"
directorio_conjunto2 = r"C:\Users\Anna Beristain\Downloads\practica2.3Vision\conjunto2"

resultados_conjunto1 = procesar_conjunto(directorio_conjunto1)
resultados_conjunto2 = procesar_conjunto(directorio_conjunto2)

columnas = ["Imagen", "Media", "Varianza", "Asimetría", "Curtosis", "Energía", "Contraste", "Entropía", "Homogeneidad"]
tabla_conjunto1 = pd.DataFrame(resultados_conjunto1, columns=columnas)
tabla_conjunto2 = pd.DataFrame(resultados_conjunto2, columns=columnas)

print("Tabla de Momentos Estadísticos - Conjunto 1:")
print(tabla_conjunto1[["Imagen", "Media", "Varianza", "Asimetría", "Curtosis"]])
print("\nTabla de Descriptores GLCM - Conjunto 1:")
print(tabla_conjunto1[["Imagen", "Energía", "Contraste", "Entropía", "Homogeneidad"]])

print("\nTabla de Momentos Estadísticos - Conjunto 2:")
print(tabla_conjunto2[["Imagen", "Media", "Varianza", "Asimetría", "Curtosis"]])
print("\nTabla de Descriptores GLCM - Conjunto 2:")
print(tabla_conjunto2[["Imagen", "Energía", "Contraste", "Entropía", "Homogeneidad"]])

promedios_conjunto1 = tabla_conjunto1.iloc[:, 1:].mean()
desviaciones_conjunto1 = tabla_conjunto1.iloc[:, 1:].std()
promedios_conjunto2 = tabla_conjunto2.iloc[:, 1:].mean()
desviaciones_conjunto2 = tabla_conjunto2.iloc[:, 1:].std()

print("\nPromedios Conjunto 1:")
print(promedios_conjunto1)
print("\nDesviaciones Estándar Conjunto 1:")
print(desviaciones_conjunto1)
print("\nPromedios Conjunto 2:")
print(promedios_conjunto2)
print("\nDesviaciones Estándar Conjunto 2:")
print(desviaciones_conjunto2)

def crear_graficos(tabla1, tabla2, columnas):
    for columna in columnas:
        plt.figure(figsize=(10, 5))

        # grafico Caja
        plt.subplot(1, 2, 1)
        plt.boxplot([tabla1[columna], tabla2[columna]], labels=["Conjunto 1", "Conjunto 2"])
        plt.title(f"grafico Caja de {columna}")
        plt.ylabel(columna)

        # dispersión
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(tabla1)), tabla1[columna], label="Conjunto 1", alpha=0.6)
        plt.scatter(range(len(tabla2)), tabla2[columna], label="Conjunto 2", alpha=0.6)
        plt.title(f"Dispersión de {columna}")
        plt.xlabel("Índice de Imagen")
        plt.ylabel(columna)
        plt.legend()

        plt.tight_layout()
        plt.show()

crear_graficos(tabla_conjunto1, tabla_conjunto2, columnas[1:])
