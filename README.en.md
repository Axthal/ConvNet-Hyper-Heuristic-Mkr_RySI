# ConvNet Hyper-Heuristic Maker - RySI
*Leer esto en [Español](https://github.com/Axthal/ConvNet-Hyper-Heuristic-Mkr_RySI/blob/master/README.md)*

Hyper-Heuristic generator of evolutionary Convolutional Neural Networks, as part of the master's thesis project in RySI, LANIA, Mexico, under supervision of Ph.D. Saúl Domínguez Isidro.

### General Description

To test the approach proposed in the thesis, a program was created to implement the Hyperheuristic algorithm that was able to create and train Neuronal Convolutive Networks for image classification through evolutionary programming and a Metaheuristic selector, with the addition of being able to write the 3 best models in plain text files (.txt) and be able to select them to read them and try other images similar to those that were trained too.

This is a console program, made with C# using the .NET Core framework, which allows it to be compiled and run for Windows, Linux or MacOS computers. It does not use any framework for Neural Networks, so it was developed from scratch using theoretical information available on Neuronal Networks, Convolutive Neural Networks and Bio-inspired Computing. The development environment used was Visual Studio 2019.

For the program coding, a combinatorial approach between structural programming and object-oriented programming was used in an attempt to maximize efficiency while retaining reusability and maintainability, using a spiral development methodology that would allow the completeness of Hyperheuristics to be achieved.

In order to reduce the evolutionary training time of each neuronal model, the program uses parallel programming through .NET Task class which behaves like a thread, so that the efficiency of each individual in the weight population can be tested. Subsequently, the entire training process of each individual (newly created neuronal model) was included in their particular Task.
