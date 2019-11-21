using System;
using CNN_RySI.CNN;
using CNN_RySI.CNN.Components;
using CNN_RySI.MLP;

namespace CNN_RySI.Structures
{
    public class EvoAIndividual
    {
        public int ID { get; set; }
        public ConvLayer[] CLayers { get; set; }
        public int[] NLayers { get; set; }
        public int Epochs { get; set; }
        public int BatchSize { get; set; }
        public double ErrorTolerance { get; set; }
        public double TotalMinutesRequired { get; set; }
        public double AccuracyPercent { get; set; }
        public double Victories { get; set; }
        public string PreHyperparameters { get; set; }
        public string[] TrainingSummary { get; set; }
        public int Generation { get; set; }
        public ConvolutionNetwork TrainedCNN { get; set; }
        public NeuralNetwork TrainedNN { get; set; }
        public EvoAIndividual(int ID, ConvLayer[] clayers, int[] nlayers, int epochs, int batchSize, double errorTolerance, int generation)
        {
            this.ID = ID;
            //Obtener las capas convolutivas (objetos)
            CLayers = new ConvLayer[clayers.Length];
            Array.Copy(clayers, CLayers, CLayers.Length);
            //Obtener las capas neuronales (números, cada número es el total de neuronas y su posición la capa)
            NLayers = new int[nlayers.Length];
            Array.Copy(nlayers, NLayers, NLayers.Length);
            //Obtener los demás parámetros
            Epochs = epochs;
            BatchSize = batchSize;
            ErrorTolerance = errorTolerance;
            Generation = generation;
        }
    }
}
