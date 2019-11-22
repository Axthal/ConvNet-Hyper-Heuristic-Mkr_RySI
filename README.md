# ConvNet Hyper-Heuristic Maker - RySI
*Read this in [English](https://github.com/Axthal/ConvNet-Hyper-Heuristic-Mkr_RySI/blob/master/README.en.md)*

Generador hiperheurístico de Redes Neuronales Convolucionales evolutivas, como parte del proyecto de tesis de maestría en RySI: IMPLEMENTACIÓN DE UNA HIPERHEURÍSTICA PARA EL DISEÑO Y ENTRENAMIENTO DE REDES NEURONALES ARTIFICIALES PARA CLASIFICACIÓN DE IMÁGENES 

Universidad LANIA, México. Desarrollado por Alejandro Cabrera Lagunes, bajo la supervisión del Dr. Saúl Domínguez Isidro.

### Descripción General

Para probar el enfoque propuesto en la tesis de grado, se realizó un programa para implementar el algoritmo Hiperheurístico que fuera capaz de crear y entrenar Redes Neuronales Convolutivas para la clasificación de imágenes mediante programación evolutiva y el uso de un selector Metaheurístico, con el añadido de poder escribir los 3 mejores modelos en archivos de texto plano (.txt) y poder seleccionarlos para leerlos y probar con otras imágenes semejantes a las que fue entrenado.

El programa es de consola, realizado con el lenguaje de programación C# usando el framework .NET Core, el cual permite que pueda ser compilado y ejecutado para computadoras con Windows, Linux o MacOS. No utiliza algún framework para Redes Neuronales, por lo que fue desarrollado desde cero usando información teórica disponible sobre Redes Neuronales y Redes Neuronales Convolutivas y sobre Cómputo Bio-inspirado. El entorno de desarrollo utilizado fue Visual Studio 2019.

Para la codificación del programa se utilizó un enfoque combinatorio entre la programación estructural y la programación orientada a objetos, en un intento de maximizar la eficiencia mientras se conserva la reusabilidad y mantenibilidad, usando una metodología espiral de desarrollo que permitiera alcanzar la completitud de la Hiperheurística. 

Con el fin de disminuir el tiempo de entrenamiento evolutivo de cada modelo neuronal, el programa se auxilia de programación en paralelo mediante el uso de la clase Task de .NET que definida de forma simple se comporta como un hilo o thread que se ejecuta de forma asíncrona, de tal forma que pueda probarse la eficiencia de cada uno de los individuos de la población de pesos. Posteriormente todo el proceso de entrenamiento de cada individuo (modelo neuronal recién creado) fue incluido en su particular Task.
