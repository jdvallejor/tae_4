# Técnicas en aprendizaje estadístico

## Cuarta entrega

### Integrantes

* ÓSCAR ALEJANDRO LÓPEZ PAZ
* DUVAN STIVEN MAYO TENJO
* NEIL YAMITH TUIRÁN TURIÁN
* SANTIAGO URIBE VELÁSQUEZ
* JUAN DAVID VALLEJO RESTREPO

### Instrucciones

Instalar la libreria scikit-learn

```sh
pip install -U scikit-learn
```

Instalar la libreria opncv

```sh
pip install opencv-python
```

Para entrenar los modelos se debe ejecutar el archivo main.py. Los modelos generados se guardan en la carpeta *models*.

Para correr el programa de reconocimiento de dígitos se debe ejecutar el archivo zip_recognition.py. La imagen que sea desea poner a prueba debe estar guardada en la raiz del proyecto. El nombre de la imagen se debe especificar en la línea 36:

~~~~~~python
cv2.imread("PUT_NAME_HERE.jpg")
~~~~~~

**Es importante que los digitos no esteń muy cerca a los borden de la imagen.** El algoritmo que busca los dígitos falla en estos casos.

En el archivo "poner nombre del pdf" se encuentra la descripción de todo el trabajo realizado.