using System;
using System.Collections.Generic;
using System.Text;
using CNN_RySI.MLP.Components;

namespace CNN_RySI.MLP.Functions
{
    class PropagationRule
    {
        /// <summary>
        /// Método con la Regla de Propagación Lineal. 
        /// Los pesos son multiplicados con los valores y sumados.
        /// </summary>
        /// <param name="Inputs">Las conexiones de entrada de la neurona</param>
        /// <returns></returns>
        public static double Lineal(Connection[] Inputs)
        {
            double sum = 0;
            for (int c = 0; c < Inputs.Length; c++)
            {
                sum += Inputs[c].Weight * Inputs[c].Value;
            }
            return sum;
        }
        /// <summary>
        /// Método con la Regla de Propagación Euclidiana. 
        /// Los valores son restados con  el correspondiente peso y exponenciados al cuadrado
        /// </summary>
        /// <param name="Inputs"></param>
        /// <returns></returns>
        public static double Euclidean(Connection[] Inputs)
        {
            double sum = 0;
            for (int c = 0; c < Inputs.Length; c++)
            {
                sum += Math.Pow(Inputs[c].Value - Inputs[c].Weight, 2);
            }
            return sum;
        }
    }
}