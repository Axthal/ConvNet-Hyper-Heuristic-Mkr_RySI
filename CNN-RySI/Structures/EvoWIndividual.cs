using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.Structures
{
    class EvoWIndividual
    {
        public int ID { get; set; }
        public double[] KValues { get; set; }
        public double[] WValues { get; set; }
        public int Victories { get; set; }
        public double Error { get; set; }
        public int NumFunction { get; set; }
        public EvoWIndividual() { }
        public EvoWIndividual(int TotalValues) { WValues = new double[TotalValues]; }
    }
}
