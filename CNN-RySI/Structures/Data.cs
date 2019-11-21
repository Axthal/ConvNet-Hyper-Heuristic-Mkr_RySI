using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.Structures
{
    public class Data
    {
        /// <summary>
        /// Atributo que almacenará los valores que componen a la imagen
        /// </summary>
        public double[] Values { get; set; }
        /// <summary>
        /// Atributo que almacenará los valores que componen a la imagen en un mapa [Dimension o Canal][Fila][Columna]
        /// </summary>
        public double[][][] Values_Jagged { get; set; }
        public int[] Expected { get; set; }
        public string[] Categories { get; set; }
        public Data(double[] Values, double[][][] Values_Jagged, int[] Expected, string[] Categories)
        {
            if (Expected == null)
                throw new System.ArgumentException("Expected cannot be null");
            if (Categories == null)
                throw new System.ArgumentException("Categories cannot be null");
            this.Values = new double[Values.Length];
            Array.Copy(Values, 0, this.Values, 0, Values.Length);

            this.Values_Jagged = new double[Values_Jagged.Length][][];
            //Por cada uno de los canales o dimensiones
            for (int ixD = 0; ixD < this.Values_Jagged.Length; ixD++)
            {
                this.Values_Jagged[ixD] = new double[Values_Jagged[ixD].Length][];
                //Por cada uno de las filas
                for (int ixR = 0; ixR < this.Values_Jagged[ixD].Length; ixR++)
                {
                    this.Values_Jagged[ixD][ixR] = new double[Values_Jagged[ixD][ixR].Length];
                    Array.Copy(Values_Jagged[ixD][ixR], this.Values_Jagged[ixD][ixR], Values_Jagged[ixD][ixR].Length);
                }
            }
            this.Expected = new int[Expected.Length];
            Array.Copy(Expected, 0, this.Expected, 0, Expected.Length);
            this.Categories = new string[Categories.Length];
            Array.Copy(Categories, 0, this.Categories, 0, Categories.Length);
        }
        public Data(double[] Values, double[][][] Values_Jagged)
        {
            this.Values = new double[Values.Length];
            Array.Copy(Values, 0, this.Values, 0, Values.Length);
            this.Values_Jagged = new double[Values_Jagged.Length][][];
            //Por cada uno de los canales o dimensiones
            for (int ixD = 0; ixD < this.Values_Jagged.Length; ixD++)
            {
                this.Values_Jagged[ixD] = new double[Values_Jagged[ixD].Length][];
                //Por cada uno de las filas
                for (int ixR = 0; ixR < this.Values_Jagged[ixD].Length; ixR++)
                {
                    this.Values_Jagged[ixD][ixR] = new double[Values_Jagged[ixD][ixR].Length];
                    Array.Copy(Values_Jagged[ixD][ixR], this.Values_Jagged[ixD][ixR], Values_Jagged[ixD][ixR].Length);
                }
            }
        }
    }
}
