using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CNN_RySI.MLP.Components;
using CNN_RySI.MLP.Functions;
using CNN_RySI.Structures;

namespace CNN_RySI.MLP
{
    public class NeuralNetwork
    {
        public static string[] Categories { get; set; }
        public Layer[] NetLayers { get; set; }
        public enum ErrorMethod { AverageCrossEntropy, MeanSquaredError }
        protected ErrorMethod ETM;
        /// <summary>
        /// Construye el modelo de Red Neuronal FeedFoward Fullyconnected sin pesos ni valores. Retorna el total de conexiones de la red
        /// </summary>
        /// <param name="Layers">La longitud es el número de capas, cada entero es el número de neuronas por capa</param>
        /// <param name="PropagationRule">Método que determinará la regla de propagación de la neurona. Ver algunos métodos en la clase -PropagationRule-</param>
        /// <param name="ActivationFunction">Método que determinará la función de activación de la neurona. Ver algunos métodos en la clase -ActivationFunction-</param>
        /// <param name="OutputFunction">Método que determinará la función de salida de la neurona. Ver algunos métodos en la clase -OutputFunction-</param>
        /// <param name="PropagationRuleLast">Método que determinará la regla de propagación de la neurona de la última capa. Ver algunos métodos en la clase -PropagationRule-</param>
        /// <param name="ActivationFunctionLast">Método que determinará la función de activación de la neurona de la última capa. Ver algunos métodos en la clase -ActivationFunction-</param>
        /// <param name="OutputFunctionLast">étodo que determinará la función de salida de la neurona de la última capa. Ver algunos métodos en la clase -OutputFunction-</param>
        /// <param name="method">Método de cálculo del error</param>
        /// <returns></returns>
        public int Build(int[] Layers, Func<Connection[], double> PropagationRule, Func<double, double> ActivationFunction, Func<double, double> OutputFunction,
            Func<Connection[], double> PropagationRuleLast, Func<double, double> ActivationFunctionLast, Func<double, double> OutputFunctionLast, ErrorMethod method)
        {
            //Architecture = string.Join(',', Layers);
            //this.UseSoftMaxOutputLayer = UseSoftMaxOutputLayer;
            ETM = method;
            int nConnections = 0;
            int nConnPerLayer;
            //Crear todas las capas
            NetLayers = new Layer[Layers.Length];
            for (int i = 0; i < Layers.Length; i++)
            {
                //Obtener el número de conexiones de entrada de cada neurona de la capa. Si es la capa 0, cada neurona tendrá 1 input. En caso contrario, tendrá el núm. de neuronas previas
                int nInputs = i == 0 ? 1 : Layers[i - 1] + 1; //El input del bias
                //Obtener el número de todas las conexiones, con el fin de devolverlo para saber el total de pesos. Si es la capa 0, el número de conexiones es 0.
                nConnPerLayer = i == 0 ? 0 : (Layers[i - 1] + 1) * Layers[i]; //Mas el input del bias (que será ponderado también)
                nConnections += nConnPerLayer;
                //Creación de un array de neuronas en la capa 'i'
                NetLayers[i] = new Layer() { Neurons = new Neuron[Layers[i]], WeightedConnections = nConnPerLayer };
                //Instanciar cada Neurona (sin pesos ni valores)
                for (int j = 0; j < Layers[i]; j++)
                {
                    if (i < Layers.Length - 1)
                        NetLayers[i].Neurons[j] = new Neuron(PropagationRule, ActivationFunction, OutputFunction, nInputs);
                    else
                        //Es la última capa, por lo que puede tener un comportamiento diferente
                        NetLayers[i].Neurons[j] = new Neuron(PropagationRuleLast, ActivationFunctionLast, OutputFunctionLast, nInputs);
                }
            }
            //NetWeights = new double[nConnections];
            return nConnections;
        }
        /// <summary>
        /// Realiza el procesado FeedForward de la Red Neuronal. Retorna el error conforme a lo esperado.
        /// </summary>
        /// <param name="Weight">Vector de pesos que se asignarán a las conexiones neuronales</param>
        /// <param name="Values">Vector de valores para capa de entrada</param>
        /// <param name="Expected">Vector con los valores esperados (1 en la posición esperada y 0 en las demás)</param>
        /// <returns></returns>
        public double Process(double[] Weight, double[] Values, int[] Expected)
        {
            int IxGlobal = 0;
            double Error = 0;
            if (NetLayers.Length < 2)
                throw new Exception("La red neuronal debe ser construida y/o tener por lo menos 2 capas");
            //En la primera capa, sólo colocar los valores en los outputs
            for (int ixD = 0; ixD < Values.Length; ixD++)
            {
                NetLayers[0].Neurons[ixD].Output = Values[ixD];
            }
            //Para cada capa a partir de la segunda
            for (int ixL = 1; ixL < NetLayers.Length; ixL++)
            {
                //En cada neurona de la capa actual...                                                             
                for (int ixN = 0; ixN < NetLayers[ixL].Neurons.Length; ixN++)
                {
                    //Colocar los valores y los pesos en el vector de Connection
                    for (int ixI = 0; ixI < NetLayers[ixL].Neurons[ixN].Inputs.Length; ixI++)
                    {
                        //Si no es el input del bias
                        if (ixI < NetLayers[ixL].Neurons[ixN].Inputs.Length - 1)
                            //Para cada salida de la neurona de la capa anterior, colocarla
                            NetLayers[ixL].Neurons[ixN].Inputs[ixI].Value = NetLayers[ixL - 1].Neurons[ixI].Output;
                        else
                            NetLayers[ixL].Neurons[ixN].Inputs[ixI].Value = 1;
                        //Y agregar el peso
                        NetLayers[ixL].Neurons[ixN].Inputs[ixI].Weight = Weight[IxGlobal];
                        IxGlobal++;
                    }
                    //Realizar el proceso de activación de la neurona
                    NetLayers[ixL].Neurons[ixN].Activation();
                }
            }
            //Obtener la salida de la última capa (OutputLayer)
            double[] outputs = NetLayers[NetLayers.Length - 1].GetOutputs();
            //Realizar el cálculo del Softmax con las salidas
            outputs = ActivationFunction.SoftMax(outputs);
            switch (ETM)
            {
                case ErrorMethod.MeanSquaredError:
                    {
                        double dif, square, expected;
                        for (int ixO = 0; ixO < outputs.Length; ixO++)
                        {
                            //Softmax precisa que la salida no esperada sea cero [0,0,1,0]
                            expected = Math.Max(0, Expected[ixO]);
                            dif = expected - outputs[ixO];
                            square = Math.Pow(dif, 2.0) * .5;
                            Error += square;
                        }
                    }
                    break;
                case ErrorMethod.AverageCrossEntropy:
                    {
                        double ln = 0;
                        double expected;
                        for (int ixO = 0; ixO < outputs.Length; ixO++)
                        {
                            expected = Math.Max(0, Expected[ixO]);
                            ln += Math.Log(outputs[ixO]) * expected;
                        }
                        Error = -(ln);
                    }
                    break;
            }
            return Error;
        }
        /// <summary>
        /// Coloca un vector de pesos dentro de las conexiones neuronales
        /// </summary>
        /// <param name="weights">Vector de pesos</param>
        public void FillWeights(double[] weights)
        {
            //Llenar en el modelo los pesos
            int ixWF = 0;
            //Por cada una de las capas
            for (int ixL = 1; ixL < NetLayers.Length; ixL++)
            {
                //por cada una de las neuronas
                for (int ixN = 0; ixN < NetLayers[ixL].Neurons.Length; ixN++)
                {
                    //Por cada uno de los inputs de la neurona
                    for (int ixC = 0; ixC < NetLayers[ixL].Neurons[ixN].Inputs.Length; ixC++)
                    {
                        NetLayers[ixL].Neurons[ixN].Inputs[ixC].Weight = weights[ixWF];
                        ixWF++;
                    }
                }
            }
        }
        /// <summary>
        /// Método auxiliar para el procesamiento de un conjunto de datos en una estructura FeedFoward (Ya entrenada)
        /// </summary>
        /// <param name="data">Imagen a la que se le encontrará una clasificación</param>
        /// <returns></returns>
        public Prediction Predict(Data data)
        {
            Prediction result = new Prediction();
            if (NetLayers.Length < 2)
                throw new Exception("La red neuronal debe ser construida y/o tener por lo menos 2 capas");
            //En la primera capa, sólo colocar los valores en los outputs
            for (int ixD = 0; ixD < data.Values.Length; ixD++)
            {
                NetLayers[0].Neurons[ixD].Output = data.Values[ixD];
            }
            //Para cada capa a partir de la segunda
            for (int ixL = 1; ixL < NetLayers.Length; ixL++)
            {
                //En cada neurona de la capa actual...
                for (int ixN = 0; ixN < NetLayers[ixL].Neurons.Length; ixN++)
                {
                    //Colocar los valores y los pesos en el vector de Connection
                    for (int ixI = 0; ixI < NetLayers[ixL].Neurons[ixN].Inputs.Length; ixI++)
                    {
                        //Para cada salida de la neurona de la capa anterior, colocarla en el vector de Inputs
                        if (ixI < NetLayers[ixL].Neurons[ixN].Inputs.Length - 1)
                            //Para cada salida de la neurona de la capa anterior, colocarla en el vector de Inputs
                            NetLayers[ixL].Neurons[ixN].Inputs[ixI].Value = NetLayers[ixL - 1].Neurons[ixI].Output;
                        else
                            NetLayers[ixL].Neurons[ixN].Inputs[ixI].Value = 1;
                    }
                    //Realizar el proceso de activación
                    NetLayers[ixL].Neurons[ixN].Activation();
                }
            }
            //Obtener la salida de la última capa (OutputLayer)
            double[] outputs = NetLayers[NetLayers.Length - 1].GetOutputs();
            //Realizar el cálculo del Softmax con las salidas
            outputs = ActivationFunction.SoftMax(outputs);
            double best = 0;
            double second = 0;
            int ixBest = -1;
            int ixSecond = -1;
            for (int ixO = 0; ixO < outputs.Length; ixO++)
            {
                if (best < outputs[ixO])
                {
                    best = outputs[ixO];
                    ixBest = ixO;
                }
            }
            for (int ixO = 0; ixO < outputs.Length; ixO++)
            {
                if (second < outputs[ixO] && outputs[ixO] < best)
                {
                    second = outputs[ixO];
                    ixSecond = ixO;
                }
            }
            if (ixBest < 0)
                ixBest = 0;
            if (ixSecond < 0)
                ixSecond = 0;
            result.BestAccuracy = best;
            result.SecondAccuracy = second;
            if (data.Categories != null)
            {
                result.BestResult = data.Categories[ixBest];
                result.SecondResult = data.Categories[ixSecond];
                result.hit = ixBest == Array.IndexOf(data.Expected, 1);
            }
            else if (Categories != null)
            {
                result.BestResult = Categories[ixBest];
                result.SecondResult = Categories[ixSecond];
                result.hit = true;
            }
            else
            {
                result.BestResult = best.ToString();
                result.SecondResult = second.ToString();
                result.hit = true;
            }
            return result;
        }
    }
}