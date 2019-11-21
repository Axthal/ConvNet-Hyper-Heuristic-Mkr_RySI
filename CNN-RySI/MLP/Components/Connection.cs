using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.MLP.Components
{
    public class Connection
    {
        /// <summary>
        /// El peso (sináptico) de la conexión
        /// </summary>
        public double Weight { get; set; }
        /// <summary>
        /// El valor que contiene esa conexión
        /// </summary>
        public double Value { get; set; }
        public Connection() { }
        public Connection(double W, double V) { Weight = W; Value = V; }
    }
}
