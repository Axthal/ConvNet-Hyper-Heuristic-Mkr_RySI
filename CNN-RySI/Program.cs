using CNN_RySI.CNN;
using CNN_RySI.CNN.Components;
using CNN_RySI.MLP;
using CNN_RySI.MLP.Functions;
using CNN_RySI.Structures;
using CNN_RySI.Tools;
using CNN_RySI.Train;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace CNN_RySI
{
    class Program
    {

        /* RUTA PARA OBTENER LAS IMÁGENES DE ENTRENAMIENTO ******************************************************/
        //Windows
        //static string TrainImages = @"C:\Users\acabreral\Documents\PERSONAL\MRYSI\Tesis\Implementación\MNIST_Numbers\train-images-idx3-ubyte\train-images.idx3-ubyte";
        //static string TrainLabels = @"C:\Users\acabreral\Documents\PERSONAL\MRYSI\Tesis\Implementación\MNIST_Numbers\train-labels-idx1-ubyte\train-labels.idx1-ubyte";
        //MAC
        //static string TrainImages = @"/Users/axthal/Documents/LANIA/Tesis/Experimentaciones/MNIST_Numbers/train-images-idx3-ubyte";
        //static string TrainLabels = @"/Users/axthal/Documents/LANIA/Tesis/Experimentaciones/MNIST_Numbers/train-labels-idx1-ubyte";
        //Windows ACER Negra
        static string TrainImages = @"C:\Users\Axthal\Documents\LANIA\MRYSI\Tesis\Experimentacion\MNIST_Numbers\train-images.idx3-ubyte";
        static string TrainLabels = @"C:\Users\Axthal\Documents\LANIA\MRYSI\Tesis\Experimentacion\MNIST_Numbers\train-labels.idx1-ubyte";
        //Windows DELL
        //static string TrainImages = @"C:\Users\Yami\Documents\Alec\MRySI\MNIST_Numbers\train-images.idx3-ubyte";
        //static string TrainLabels = @"C:\Users\Yami\Documents\Alec\MRySI\MNIST_Numbers\train-labels.idx1-ubyte";
     

        /* RUTA PARA OBTENER LAS IMÁGENES DE PRUEBA *************************************************************/
        //Windows
        //static string TestImages = @"C:\Users\acabreral\Documents\PERSONAL\MRYSI\Tesis\Implementación\MNIST_Numbers\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte";
        //static string TestLabels = @"C:\Users\acabreral\Documents\PERSONAL\MRYSI\Tesis\Implementación\MNIST_Numbers\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte";
        //MAC
        //static string TestImages = @"/Users/axthal/Documents/LANIA/Tesis/Experimentaciones/MNIST_Numbers/t10k-images-idx3-ubyte";
        //static string TestLabels = @"/Users/axthal/Documents/LANIA/Tesis/Experimentaciones/MNIST_Numbers/t10k-labels-idx1-ubyte";
        //Windows ACER
        static string TestImages = @"C:\Users\Axthal\Documents\LANIA\MRYSI\Tesis\Experimentacion\MNIST_Numbers\t10k-images.idx3-ubyte";
        static string TestLabels = @"C:\Users\Axthal\Documents\LANIA\MRYSI\Tesis\Experimentacion\MNIST_Numbers\t10k-labels.idx1-ubyte";
        //Windows DELL
        //static string TestImages = @"C:\Users\Yami\Documents\Alec\MRySI\MNIST_Numbers\t10k-images.idx3-ubyte";
        //static string TestLabels = @"C:\Users\Yami\Documents\Alec\MRySI\MNIST_Numbers\t10k-labels.idx1-ubyte";
       
        /* RUTA PARA LOS MODELOS ENTRENADOS **********************************************************************/
        //Windows
        //static string CarpetaModelos = @"C:\Users\acabreral\Documents\PERSONAL\MRYSI\Tesis\Implementación\Modelos\";
        //MAC
        //static string CarpetaModelos = @"/Users/axthal/Documents/LANIA/Tesis/Experimentaciones/Modelos/";
        //Windows ACER
        static string CarpetaModelos = @"C:\Users\Axthal\Documents\LANIA\MRYSI\Tesis\Experimentacion\Modelos\";
        //Windows DELL
        //static string CarpetaModelos = @"C:\Users\Yami\Documents\Alec\Modelos\";
        

        static void Main(string[] args)
        {
            Train_CNN_Hyperheuristic();
        }
        public static void Train_CNN_Hyperheuristic()
        {
            #region 1. LECTURA DE DATOS DE ENTRENAMIENTO
            Console.WriteLine($"LECTURA DE DATOS DE ENTRENAMIENTO: {DateTime.Now}");
            int mainCategorie = 0; //Buscará números 0 (clase)
            Data[] DataSetTrain = DataHelper.ImportData_MNIST_TrainOneCat(TrainImages, TrainLabels, mainCategorie, 1500);
            //Es imprecindible que la categoría principal deba ir al inicio
            //Aquí se leerán N cantidad de imágenes del número 3 y de los que sigan
            int[] Categories = { mainCategorie, 3 };
            Data[] DataSetTest = DataHelper.ImportData_MNIST_TestOneCat(TestImages, TestLabels, Categories, 500);
            #endregion
            #region 2. INVOCAR AL MÉTODO DE ENTRENAMIENTO DE ARQUITECTURAS
            Console.WriteLine($"COMIENZO DEL ENTRENAMIENTO: {DateTime.Now}");
            EvoAIndividual[] BestArchs = Trainers.Hyperheuristic_Trainning(DataSetTrain, DataSetTest, Categories.Length);
            #endregion
            #region 3. MOSTRAR LOS RESULTADOS
            Console.WriteLine("RESULTADOS:");
            for (int ixR = 0; ixR < 4; ixR++)
            {
                string data = DataHelper.GetArchIndividualData(BestArchs[ixR]);
                Console.WriteLine($"{ixR + 1}° LUGAR");
                Console.WriteLine(data);
                string[] Summary = BestArchs[ixR].TrainingSummary;
                string summ = "";
                for (int ixS = 0; ixS < Summary.Length; ixS++)
                {
                    summ += " - " + Summary[ixS];
                    Console.WriteLine($"{Summary[ixS]}");
                }
                Console.WriteLine($"GUARDANDO EL MODELO COMPLETO EN UN TXT...");
                string Fecha = $"{DateTime.Now.ToString("dd-MM-yyyy_HH_mm")}";
                DataHelper.WriteConvolutionalNeuralNetworkModel(BestArchs[ixR].TrainedCNN, BestArchs[ixR].TrainedNN, CarpetaModelos, $"Hiperheuristica_{Fecha}_{ixR + 1}_Lugar", $"{data} ---- {summ}");
                Console.WriteLine("");
            }
            Console.WriteLine("LOS MODELOS FUERON GUARDADOS EN LA RUTA " + CarpetaModelos);
            Console.WriteLine("PRESIONE 'ENTER' PARA FINALIZAR EL PROGRAMA");
            Console.ReadLine();
            #endregion
        }
    }
}
