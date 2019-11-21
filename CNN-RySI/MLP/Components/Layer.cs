using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CNN_RySI.MLP.Components
{
    public class Layer
    {
        /// <summary>
        /// Vector de Neuronas que contiene la capa
        /// </summary>
        public Neuron[] Neurons { get; set; }
        public int WeightedConnections { get; set; }
        /// <summary>
        /// Método que obtiene un vector con los valores de salida de todas las neuronas de la capa, mediante Linq
        /// </summary>
        /// <returns></returns>
        public double[] GetOutputsLinq()
        {
            return Neurons.Select(n => n.Output).ToArray();
        }
        /// <summary>
        /// Método que obtiene un vector con los valores de salida de todas las neuronas de la capa, de forma más general
        /// </summary>
        /// <returns></returns>
        public double[] GetOutputs()
        {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
                outputs[i] = Neurons[i].Output;
            return outputs;
        }
    }
}
