using System;
using System.Text;

public class EvoFunction {
    public double Hits { get; set; }
    public double Calls { get; set; }
    public EvoFunction() {
        //Asegurando el 50% inicial y de ahí algo más del 0%
        Hits = 1; Calls = 2;
    }
    public double GetAccuracy() {
        return Hits / Calls;
    }
}