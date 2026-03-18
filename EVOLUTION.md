# 📈 Log de Experimentos: ResNet-50 en CIFAR-10

Este documento sirve como registro de ingeniería para documentar la evolución del modelo, las hipótesis probadas y los resultados obtenidos en cada iteración del proyecto.

---

## Tabla Comparativa de Resultados

| ID | Description | Epochs | Train Acc | Dev Acc | Test Acc | State |
|:---|:---|:---:|:---:|:---:|:---|:---|
| 01 | **Baseline:** ResNet-50 standard | 20 | 0.9546  | 0.6380 | 0.6378 | Overfitting |
| 02 | **Data Augmentation:** Flip + Rotation | 20 | 0.8065  | 0.7517  | 0.7517 | High bias |  
| 03 | **Train more time:** More epochs + early stopping | 28 | 0.8986 | 0.7799  | 0.7799 | Overfitting |  
| 04 | **Scale images, variable LR:** Images 160x160 + ReduceLROnPlateau + DropOut | 33 | 0.9850 | 0.8928  | - | Overfitting |  

---

## Detalle de Iteraciones

Para poder entrenar el modelo con los datos de CIFAR-10 usando la arquitectura de ResNet-50 (la cual es muy compleja) se tuvo que implementar un reescalado de 32x32 a 64x64 píxeles. Sin esta transformación, la profundidad de la ResNet-50 colapsaba los mapas de características a 1x1, impidiendo el aprendizaje de patrones espaciales complejos en clases visualmente similares.

---

### Experimento #01
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (LR inicial: 0.0001)
    * **Batch Size:** 64
    * **Normalización:** Píxeles reescalados a [0, 1].
    * **Epochs:** 20

#### Resultados
* **Training Accuracy:** 95.46%
* **Dev Accuracy:** 63.80%
* **Test Accuracy:** 63.78%

<br>

![trainingphoto](/images/1exp.png)

#### Conclusiones

1.  **Overfitting:** Se observa un gap de más del 32% entre el entrenamiento y el test. El modelo esta memorizando el ruido del set de entrenamiento en lugar de generalizar características.
2.  **Inestabilidad en Validación:** A pesar de usar un LR conservador de 0.0001, las curvas de *val_accuracy* muestran inestabilidad y los saltos entre epocas son muy grandes. Intentaremos reducir el LR para suavizar la curva.
3. **Conclusion:** El modelo no esta generalizando correctamente y su comportamiento es pobre.
3.  **Mejoras para el siguiente experimento:** 
- Reducir el LR de 0.0001 a 0.00005
- Aplicar data augmentation con rotacion y zoom aleatorios a las imagenes del set para no sobreajustar a los datos de entrenamiento.





---

### Experimento #02
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (LR inicial: 0.00005)
    * **Batch Size:** 64
    * **Normalización:** Píxeles reescalados a [0, 1].
    * **Epochs:** 20


#### Resultados
* **Training Accuracy:** 80.65%
* **Dev Accuracy:** 75.17%
* **Test Accuracy:** 75.17%

<br>

![trainingphoto](/images/2exp.png)

#### Conclusiones

1.  **Overfitting:** Vemos como mejoro la accuracy en el devset y el gap con train acc es de 5%, esto sugiere que el modelo esta generalizando mucho mejor y no sobreajuistando a los datos de entrenamiento.
2.  **Inestabilidad en Validación:** La reduccion del LR surtio efecto y la curva de accuracy tiene un comportamiento mas estable sin tantos saltos abruptos. 
3.  **Conclusion:** El modelo generalizo mucho mejor en el set de test, vemos que la accuracy en validation sigue subiendo por lo que hay margen de mejora. 
4.  **Mejoras para el siguiente experimento:** 
- Continuar el entrenamiento por mas epochs
- Implementar early-stopping para parar el punto justo donde el modelo tiene mejor performance



---

### Experimento #03
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (LR inicial: 0.00005)
    * **Batch Size:** 64
    * **Normalización:** Píxeles reescalados a [0, 1].
    * **Epochs:** 28


#### Resultados
* **Training Accuracy:** 89.86%
* **Dev Accuracy:** 77.91%
* **Test Accuracy:** 77.99%

<br>

![trainingphoto](/images/3exp.png)

Se recupero el entrenamiento de las 20 epocas anteriores para no tener que reentrenar el modelo y se sigio por 20 epocas mas aplicando early-stopping con una paciencia de 7 monitoreando la val_loss para darle tiempo al modelo de mejorar antes de cortar el entrenamiento. 

#### Conclusiones

1.  **Overfitting:** Si bien el modelo mejoro su performance muy levemente en dev y test, vemos que la accuracy en train estaba subiendo muy rapido respecto a dev por lo que el modelo esta sobreajustando nuevamente.
3.  **Conclusion:**  El modelo sobreajusto mucho y el early stopping corto el entrenamiento, regularizar o aumentar data favoreceria el desempeño
4.  **Mejoras para el siguiente experimento:** 
- Reescalado de imagenes a 160 x 160
- ReduceLROnPlateau para el LR
- DropOut


---


### Experimento #04
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (160, 160, 3)
    * **Optimizer:** Adam (LR inicial: 1.0000e-04)
    * **Batch Size:** 64
    * **Normalización:** Píxeles reescalados a [0, 1].
    * **DropOut:** 0.5 on dense final layer
    * **Epochs:** 33


#### Resultados
* **Training Accuracy:** 98.50%
* **Dev Accuracy:** 89.28%
* **Test Accuracy:** -

<br>

![trainingphoto](/images/4exp.png)


#### Conclusiones

1.  **Overfitting:** 
3.  **Conclusion:** Problema con limitaciones de recursos, agregare model checkpints para no perder el progreso en caso de error
weight decay
SGD
model checkpoint
train more epochs



---


### Experimento #05

en la epoca 40 train acc sea de 94% y dev acc de 87.1%

![trainingphoto](/images/5exp-primer.png)


--- 

### Experimento #06

Continuo el train del exp anterior hasta 80 epochs y agrego data augmentation
 layers.RandomTranslation(0.1, 0.1), # Mueve la imagen para que no memorice la posición
            layers.RandomZoom(0.1),            # Zoom aleatorio para variar escalas
            layers.RandomContrast(0.1),        # Variación de iluminación





















---