using System;
using System.Linq;

namespace CNN_RySI.MLP.Functions
{
    class ActivationFunction
    {
        /// <summary>
        /// Método con la Función Activación 'Identidad'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double Identity(double x)
        {
            return x;
        }
        /// <summary>
        /// Método con la Función Activación 'Binaria'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double Binary(double x)
        {
            return (x >= 0) ? 1 : 0;
        }
        /// <summary>
        /// Método con la Función Activación 'Binaria ampliada'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double BinaryWide(double x)
        {
            return (x >= 0) ? 1 : -1;
        }
        /// <summary>
        /// Método con la Función Activación 'Sigmoide'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        /// <summary>
        /// Método con la Función Activación 'Hiperbólica'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double Hiperbolic(double x)
        {
            return Math.Tanh(x);
        }
        /// <summary>
        /// Método con la Función Activación 'ReLU'
        /// </summary>
        /// <param name="x">Valor obtenido de la regla de propagación</param>
        /// <returns></returns>
        public static double ReLu(double x)
        {
            return Math.Max(0, x);
        }
        /// <summary>
        /// Función SoftMax.
        /// Adaptada de la versión Linq encontrada en: https://gist.github.com/jogleasonjr/55641e503142be19c9d3692b6579f221, con el fin de utilizar elementos más genéricos
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double[] SoftMax(double[] values)
        {
            //Vector temporal para almacenar los exponentes por 'e'
            double[] zexp = new double[values.Length];
            //Vector para los valores finales
            double[] softmaxV = new double[values.Length];
            double sum_zexp = 0;
            for (int i = 0; i < values.Length; i++)
            {
                //Elevar cada uno de los valores por 'e'
                zexp[i] = Math.Exp(values[i]);
                sum_zexp += zexp[i];
            }
            //Obtener los valores softmax
            for (int i = 0; i < values.Length; i++)
                softmaxV[i] = zexp[i] / sum_zexp;
            return softmaxV;
        }
    }
}
