using System;

namespace CNN_RySI.MLP.Functions
{
    class OutputFunction
    {
        /// <summary>
        /// Método con la Función de Salida Lineal. El valor que entra sale igual.
        /// </summary>
        /// <param name="y">El resultado de la activación</param>
        /// <returns></returns>
        public static double Lineal(double y)
        {
            return y;
        }
    }
}