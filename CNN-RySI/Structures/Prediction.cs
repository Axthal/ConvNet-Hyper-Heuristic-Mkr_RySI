using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.Structures
{
    public class Prediction
    {
        public double BestAccuracy { get; set; }
        public string BestResult { get; set; }
        public bool hit { get; set; }
        public double SecondAccuracy { get; set; }
        public string SecondResult { get; set; }
    }
}
