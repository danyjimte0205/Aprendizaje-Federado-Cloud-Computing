# Cloud Computing - Actividad: Aprendizaje Federado

Este repositorio contiene el trabajo realizado para la materia de Computación en la Nube, donde se implementa una simulación de aprendizaje federado utilizando el dataset MNIST. La actividad consiste en entrenar modelos locales en diferentes computadoras (una por cada miembro del equipo) y modelarlos y evaluarlos mediante distintos algoritmos: FedAvg, FedMedian y FedTrimmedMean.

## Requisitos

- Python 3.10 o superior
- TensorFlow 2.0 o superior
- scikit-learn
- matplotlib
- tqdm
- numpy

## Datos Utilizados

- **Fuente**: Dataset MNIST a través de TensorFlow

Para descargar y usar los datos, ejecuta el siguiente código en tu entorno:

```python
from tensorflow.keras.datasets import mnist
train, test = mnist.load_data()
```

## Estructura del Proyecto

* **TheModel.py:** Define el modelo de red neuronal usado por los clientes y el modelo global.
* **local_training_n.ipynb:** Plantilla de entrenamiento de los modelos localmente, cada uno con una partición diferente del conjunto de datos.
* **global_model.ipynb:** Realiza la agregación de modelos utilizando FedAvg, FedMedian y FedTrimmedMean.
* **lmodel_0.keras, ..., lmodel_4.keras:** Modelos locales entrenados.


