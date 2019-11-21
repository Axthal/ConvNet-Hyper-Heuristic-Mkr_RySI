using System;
namespace CNN_RySI.Tools
{
    public class RandomHelper
    {
        /// <summary>
        /// Genera un número aleatorio entre 2 doubles
        /// </summary>
        /// <param name="min">Número mínimo del double aleatorio</param>
        /// <param name="max">Número máximo del double aleatorio (nunca se elegirá este número)</param>
        /// <param name="decimals">Número de decimales del double generado</param>
        /// <returns></returns>
        public static double GetRandomDouble(double min, double max, int decimals)
        {
            Random r = new Random();
            return Math.Round(r.NextDouble() * (max - min) + min, decimals);
        }
        /// <summary>
        /// Genera un número aleatorio entre 2 doubles
        /// </summary>
        /// <param name="r">Objeto de tipo Random generado previamente</param>
        /// <param name="min">Número mínimo del double aleatorio</param>
        /// <param name="max">Número máximo del double aleatorio (nunca se elegirá este número)</param>
        /// <param name="decimals">Número de decimales del double generado</param>
        /// <returns></returns>
        public static double GetRandomDouble(Random r, double min, double max, int decimals)
        {
            return Math.Round(r.NextDouble() * (max - min) + min, decimals);
        }
        /// <summary>
        /// Genera un número aleatorio entre 2 doubles
        /// </summary>
        /// <param name="min">Número mínimo del double aleatorio</param>
        /// <param name="max">Número máximo del double aleatorio (nunca se elegirá este número)</param>
        /// <returns></returns>
        public static double GetRandomDouble(double min, double max)
        {
            Random r = new Random();
            return r.NextDouble() * (max - min) + min;
        }
        /// <summary>
        /// Genera un número aleatorio entre 2 doubles
        /// </summary>
        /// <param name="r">Objeto de tipo Random generado previamente</param>
        /// <param name="min">Número mínimo del double aleatorio</param>
        /// <param name="max">Número máximo del double aleatorio (nunca se elegirá este número)</param>
        /// <returns></returns>
        public static double GetRandomDouble(Random r, double min, double max)
        {
            return r.NextDouble() * (max - min) + min;
        }
        /// <summary>
        /// Obtiene un número entero entre el rango otorgado, considerando el inicio y el final - 1 
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static int GetRandomInt(int min, int max)
        {
            Random r = new Random();
            return r.Next(min, max);
        }
        /// <summary>
        /// Obtiene un número entre el rango otorgado, considerando el inicio y el final
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static int GetRandomIntAbsolute(int min, int max)
        {
            Random r = new Random();
            return r.Next(min, max + 1);
        }
        /// <summary>
        /// Genera un 'Flipcoin' o tiro de moneda aleatorio
        /// </summary>
        /// <returns></returns>
        public static bool GetRandomBoolCoin()
        {
            Random r = new Random();
            return r.Next(100) >= 50;
        }
        /// <summary>
        /// Método que recorre un vector y verifica si cada valor está dentro del rango proporcionado, reemplazándolo en caso contrario con el valor más cercano del rango
        /// </summary>
        /// <param name="x">Vector a verificar</param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static double[] Reflexion(double[] x, double min, double max)
        {
            for (int i = 0; i < x.Length; i++)
            {
                while (x[i] < min || x[i] > max)
                {
                    x[i] = (x[i] < min) ? (2 * min - x[i]) : x[i];
                    x[i] = (x[i] > max) ? (2 * max - x[i]) : x[i];
                }
            }
            return x;
        }
    }
}
