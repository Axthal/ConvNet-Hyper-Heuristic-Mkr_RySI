using CNN_RySI.CNN;
using CNN_RySI.CNN.Components;
using CNN_RySI.MLP;
using CNN_RySI.MLP.Functions;
using CNN_RySI.Structures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace CNN_RySI.Tools
{
    class DataHelper
    {
        #region SECCIÓN DE LECTURA DE IMÁGENES
        /// <summary>
        /// Momento en que se está accediendo a las imágenes
        /// </summary>
        public enum DataSetMoments { Train, Validation, Testing }
        /// <summary>
        /// Método que presupone una carpeta de imágenes donde todas ellas están dentro de subcarpetas, que serán tomadas como clases. 
        /// Convertirá las imágenes en un vector de objetos tipo 'Data' (No finalizado ni probado. Todas las imágenes deben ser del mismo tamaño: ancho y alto)
        /// </summary>
        /// <param name="folderPath">Ruta de la carpeta</param>
        /// <param name="moment">Momento en que se está accediendo a las imágenes</param>
        /// <param name="sizeImage">Tamaño de las imágenes</param>
        /// <param name="totalClasses">Total de clasificaciones encontradas</param>
        /// <returns></returns>
        public static Data[] ImportData_Images(string folderPath, DataSetMoments moment, out int sizeImage, out int totalClasses)
        {
            Data[] D;
            sizeImage = 0;
            totalClasses = 0;
            string[] categories = null;
            int[] expected = null;
            string[] imgPaths = null;
            bool isTesting = moment.Equals(DataSetMoments.Testing);
            if (!isTesting)
            {
                #region --------------- PROCESAR PARA IMÁGENES DE TRAINNING Y VALIDATION --------------------
                //Obtener las rutas y los nombres de las carpetas para la clasificación
                var subdirs = Directory.GetDirectories(folderPath)
                                .Select(p => new
                                {
                                    Path = p,
                                    Name = Path.GetFileName(p)
                                }).ToArray();
                //Obtener el vector de categorías (subfolders)
                categories = subdirs.OrderBy(s => s.Name).Select(s => s.Name).ToArray();
                totalClasses = categories.Length;
                //Obtener el número total de archivos de todas las categorías (subfolders)
                int TImages = Directory.GetDirectories(folderPath)
                                .Select(p => Directory.GetFiles(p).Length).Sum();
                //Crear el vector de Data (que almacenará los datos de cada imagen)
                D = new Data[TImages];
                //Crear el vector de expected (de la misma longitud de categorías)
                expected = new int[categories.Length];
                for (int ixE = 0; ixE < categories.Length; ixE++)
                    expected[ixE] = -1;
                //Por cada una de las carpetas...
                string[] subFolderPaths = subdirs.OrderBy(s => s.Name).Select(s => s.Path).ToArray();
                for (int ixSF = 0; ixSF < subFolderPaths.Length; ixSF++)
                {
                    //...obtener todas las rutas de las imágenes
                    imgPaths = Directory.GetFiles(subFolderPaths[ixSF])
                        .Where(sp => sp.EndsWith(".png") || sp.EndsWith(".jpg") || sp.EndsWith(".jpeg") || sp.EndsWith(".bmp")).ToArray();
                    //Para cada ruta de imagen
                    //En la posición de la carpeta (categoría), poner un 1. Todos los demás son 0.
                    expected[ixSF] = 1;
                    for (int ixP = 0; ixP < imgPaths.Length; ixP++)
                    {
                        int TI = ProcessImage(ixP);
                        if (sizeImage < TI)
                            sizeImage = TI;
                    }
                }
                #endregion
            }
            else
            {
                #region--------------- PROCESAR PARA IMÁGENES DE TESTING (SIN CLASIFICACIÓN NI VALORES ESPERADOS) --------------------
                //Obtener todas las rutas de las imágenes
                imgPaths = Directory.GetFiles(folderPath)
                    .Where(sp => sp.EndsWith(".png") || sp.EndsWith(".jpg") || sp.EndsWith(".jpeg") || sp.EndsWith(".bmp")).ToArray();
                //Crear el vector de Data con el total de las rutas de imágenes
                D = new Data[imgPaths.Length];
                //Para cada ruta de imagen
                for (int ixP = 0; ixP < imgPaths.Length; ixP++)
                {
                    int TI = ProcessImage(ixP);
                    if (sizeImage < TI)
                        sizeImage = TI;
                }
                #endregion

            }
            //Método local para procesar la imagen
            int ProcessImage(int ixP)
            {
                //Obtener un mapeado de la imagen
                Bitmap img = new Bitmap(imgPaths[ixP]);
                int[] values = new int[img.Width * img.Height];
                //Cuando sea el momento, tener en cuenta de que el jagged array es de 3 dimensiones [Canal][Fila][Columna], en el cual el canal es RGB
                //int[][] valJagg = GetJaggedArray_FromVector(values);
                //int ixV = 0;
                //for (int x = 0; x < img.Width; x++)
                //{
                //    for (int y = 0; y < img.Height; y++)
                //    {
                //        Color pixelColor = img.GetPixel(x, y);
                //        values[ixV] = pixelColor.ToArgb();
                //    }
                //}
                //if (!isTesting)
                //    D[ixGlobalD] = new Data(values, valJagg, expected, categories);
                //else
                //    D[ixGlobalD] = new Data(values, valJagg);
                return values.Length; //Devolver el tamaño de la imagen
            }
            return D;
        }
        /// <summary>
        /// Método que lee una determinada cantidad de imágenes de un formato igual al dataset MNSIT
        /// </summary>
        /// <param name="filePathImgs">Ruta del archivo con las imágenes</param>
        /// <param name="filePathLbls">Ruta del archivo con las etiquetas</param>
        /// <param name="categories">Vector de categorías que serán leídas</param>
        /// <param name="maxImagesPerCat">Máximo de imágenes que serán leídas por cada categoría</param>
        /// <param name="totalClasses">Número de clases obtenidas</param>
        /// <returns></returns>
        public static Data[] ImportData_MNIST(string filePathImgs, string filePathLbls, string[] categories, int maxImagesPerCat, out int totalClasses)
        {
            Data[] D = MnistReader.ReadData(filePathImgs, filePathLbls, categories, maxImagesPerCat);
            for (int ixO = 0; ixO < 10; ixO++)
            {
                Console.WriteLine($"TOTAL {ixO}: {D.Where(d => d.Expected[ixO] == 1).Count()}");
                Random r = new Random();
                WriteImageConsole(D.Where(d => d.Expected[ixO] == 1).OrderBy(d => r.Next()).FirstOrDefault().Values_Jagged);
            }
            totalClasses = categories.Length; //10; //Del 0 al 9
            return D;
        }
        /// <summary>
        /// Método que lee una determinada cantidad de imágenes (de dos clases) de un formato igual al dataset MNSIT, para entrenamiento.
        /// </summary>
        /// <param name="filePathImgs">Ruta del archivo con las imágenes</param>
        /// <param name="filePathLbls">Ruta del archivo con las etiquetas</param>
        /// <param name="cat">Categoría principal en la que se entrenará el modelo, con el fin de marcarla con el valor esperado, y viceversa con las demás</param>
        /// <param name="maxImagesPerCat">Máximo de imágenes que serán leídas por cada categoría</param>
        /// <returns></returns>
        public static Data[] ImportData_MNIST_TrainOneCat(string filePathImgs, string filePathLbls, int cat, int maxImagesPerCat)
        {
            Data[] D = MnistReader.ReadData_TrainOneCat(filePathImgs, filePathLbls,cat, maxImagesPerCat);
            for (int ixO = 0; ixO < 10; ixO++)
                WriteImageConsole(D[ixO].Values_Jagged);
            return D;
        }
        /// <summary>
        /// Método que lee una determinada cantidad de imágenes (de dos clases) de un formato igual al dataset MNSIT, para prueba.
        /// </summary>
        /// <param name="filePathImgs">Ruta del archivo con las imágenes</param>
        /// <param name="filePathLbls">Ruta del archivo con las etiquetas</param>
        /// <param name="cats">Vector con las categorías de imágenes que serán leídas</param>
        /// <param name="maxImagesPerCat">Máximo de imágenes que serán leídas por cada categoría</param>
        /// <returns></returns>
        public static Data[] ImportData_MNIST_TestOneCat(string filePathImgs, string filePathLbls, int[] cats, int maxImagesPerCat)
        {
            Data[] D = MnistReader.ReadData_TestOneCat(filePathImgs, filePathLbls, cats, maxImagesPerCat);
            return D;
        }
        /// <summary>
        /// Método que plasma la imagen en consola de manera uniforme: Espacio vacío si el pixel es cero, # en otro caso
        /// </summary>
        /// <param name="img">La imagen en un arreglo tipo jagged</param>
        public static void WriteImageConsole(double[][][] img)
        {
            for (int ixD = 0; ixD < img.Length; ixD++)
            {
                Console.WriteLine($"DIMENSION {ixD}");
                for (int ixR = 0; ixR < img[ixD].Length; ixR++)
                {
                    int cols = img[ixD][ixR].Length;
                    for (int ixC = 0; ixC < cols; ixC++)
                    {
                        string val = img[ixD][ixR][ixC] == 0 ? " " : "#";
                        Console.Write(val);
                    }
                    Console.WriteLine();
                }
            }
        }
        #endregion
        #region SECCIÓN DE PARSING
        /// <summary>
        /// Método que obtiene una cadena de texto con la descripción del individuo evolutivo de arquitectura generado en la hiperheurística
        /// </summary>
        /// <param name="evoIndiv">El individuo evolutivo de arquitectura</param>
        /// <returns></returns>
        public static string GetArchIndividualData(EvoAIndividual evoIndiv)
        {
            string convLayers = "";
            for (int ixC = 0; ixC < evoIndiv.CLayers.Length; ixC++)
            {
                convLayers += $"({evoIndiv.CLayers[ixC].GetLayerArchitecture()});";
            }
            string data = $"Accuracy: {evoIndiv.AccuracyPercent}; " +
                $"|MinutesRequired: {evoIndiv.TotalMinutesRequired}; " +
                $"|BatchSize: {evoIndiv.BatchSize}; " +
                $"|Epochs: {evoIndiv.Epochs}; " +
                $"|ErrorTolerance: {evoIndiv.ErrorTolerance}; " +
                $"|ConvolSection=> TotalLayers: {evoIndiv.CLayers.Length}; {convLayers} " +
                $"|NeuralSection=> TotalLayers: {evoIndiv.NLayers.Length}; ({string.Join(',', evoIndiv.NLayers)}); " +
                $"|HyperparamsPreTrainned: {evoIndiv.PreHyperparameters}; " +
                $"|VictoriesLastGen: {evoIndiv.Victories}; " +
                $"|Generation: {evoIndiv.Generation}";
            return data;
        }
        /// <summary>
        /// Método que obtiene un arreglo tipo jagged a partir de un vector
        /// </summary>
        /// <param name="values">Vector con los valores</param>
        /// <param name="totalDimensions">Cantidad de dimensiones</param>
        /// <returns></returns>
        public static double[][][] GetJaggedArray_FromVector(double[] values, int totalDimensions)
        {
            double[][][] image = new double[totalDimensions][][];
            int totalValues = values.Length;
            double totalVPerDimension = totalValues / totalDimensions;
            double kSize = Math.Sqrt(totalVPerDimension);
            int ixV = 0;
            //Por cada una de las dimensiones
            for (int ixD = 0; ixD < totalDimensions; ixD++)
            {
                image[ixD] = new double[(int)kSize][];
                //Por cada una de las filas
                for (int ixR = 0; ixR < (int)kSize; ixR++)
                {
                    image[ixD][ixR] = new double[(int)kSize];
                    //Por cada una de las columnas
                    for (int ixC = 0; ixC < (int)kSize; ixC++)
                    {
                        image[ixD][ixR][ixC] = values[ixV];
                        ixV++;
                    }
                }
            }
            return image;
        }
        /// <summary>
        /// Método que obtiene un vector de valores a partir de un arreglo tipo jagged (para 'aplanado')
        /// </summary>
        /// <param name="jagged">El arreglo tipo jagged</param>
        /// <returns></returns>
        public static double[] GetVector_FromJaggedArray(double[][][] jagged)
        {
            //Obtener el número de canales o dimensiones
            int d = jagged.Length;
            //Obtener el número de filas (y por ende de columnas)
            int w = jagged[0].Length;
            //Obtener el total de valores de una sola dimensión (mapa)
            int l = w * w;
            //Vector único de todos los valores
            double[] vectorImg = new double[l * d];
            int ixValDim = 0;
            //Por cada una de las dimensiones
            for (int ixD = 0; ixD < d; ixD++)
            {
                int wRow = 0, wCol = 0;
                //Por cada uno de los valores de esa dimensión
                for (int ixM = 0; ixM < l; ixM++)
                {
                    //Colocar el valor de la imagen
                    vectorImg[ixM + ixValDim] = jagged[ixD][wRow][wCol];
                    wCol++;
                    if (wCol == w)
                    {
                        wRow++;
                        wCol = 0;
                    }
                }
                ixValDim += l;
            }
            return vectorImg;
        }
        #endregion
        #region SECCIÓN DE LECTURA / ESCRITURA DE PESOS
        static readonly string CONVOLUTION = "CONVOLUTION";
        static readonly string FULLY = "FULLYCONNECTED";
        static readonly string KERNEL = "KERNEL";
        /// <summary>
        /// Escribe un modelo RNC en un archivo .txt (arquitectura y pesos) con un formato particular (Aún no totalmente automatizado, pues faltan de incluir ciertos parámetros dentro de la estructura del archivo)
        /// </summary>
        /// <param name="CNNModel">Sección convolutiva del modelo</param>
        /// <param name="NNModel">Sección neuronal del modelo</param>
        /// <param name="FilePath">Ruta de la carpeta donde se almacenará el modelo</param>
        /// <param name="Name">Nombre del archivo (sin extensión)</param>
        /// <param name="Observations">Texto que se incorporará al inicio del archivo a modo de comentario</param>
        public static void WriteConvolutionalNeuralNetworkModel(ConvolutionNetwork CNNModel, NeuralNetwork NNModel, string FilePath, string Name, string Observations)
        {
            //FALTA QUE GUARDE EL TAMAÑO Y DIMENSIÓN EN LA QUE FUE ENTRENADA, ADEMÁS DEL NÚMERO DE CLASES QUE RECONOCE (ACTUALMENTE DEBE SER ESPECIFICADO AL LEER)
            //Escribir en un archivo txt la arquitectura del modelo, al igual que los pesos utilizados
            string fp = Path.Combine(FilePath, Name);
            //Si existe previamente un archivo como Modelo_A.txt, colocarlo en Modelo_A(1).txt
            int fileCount = -1;
            do
                fileCount++;
            while (File.Exists(fp + (fileCount > 0 ? "(" + fileCount.ToString() + ").txt" : ".txt")));
            var F = File.Create(fp + (fileCount > 0 ? "(" + (fileCount).ToString() + ").txt" : ".txt"));
            using (var W = new StreamWriter(F))
            {
                W.WriteLine("//" + Observations);
                //Escribir la arquitectura Convolucional (Número de Capas - Número de Kernels, Tamaño de Kernels, Stride, Padding, Tamaño Pooling, Stride Pooling)
                ConvLayer[] convLayers = CNNModel.GetLayers();
                //Por cada una de las capas convolucionales
                W.WriteLine(CONVOLUTION);
                for (int ixC = 0; ixC < convLayers.Length; ixC++)
                {
                    W.WriteLine($@"//Capa convolucional {ixC}");
                    W.WriteLine($"{convLayers[ixC].GetTotalKernels()};{convLayers[ixC].GetKernelSize()};{convLayers[ixC].GetKernelStride()};{convLayers[ixC].GetPadding()};{convLayers[ixC].GetPoolingSize()};{convLayers[ixC].GetPoolingStride()}");
                    //Escribir los valores de kernels - [Número de Kernel][Dimensión][Fila][Columna]
                    double[][][][] kernelValues = convLayers[ixC].GetKernelValues();
                    //Por cada uno de los Kernels
                    for (int ixK = 0; ixK < kernelValues.Length; ixK++)
                    {
                        W.WriteLine($@"//Kernel {ixK}");
                        W.WriteLine(KERNEL);
                        //Por cada una de las dimensiones
                        for (int ixD = 0; ixD < kernelValues[ixK].Length; ixD++)
                        {
                            W.WriteLine($@"//Dimension {ixD}");
                            string vals = "";
                            //Por cada una de las filas
                            for (int ixRow = 0; ixRow < kernelValues[ixK][ixD].Length; ixRow++)
                            {
                                //Por cada una de las columnas
                                for (int ixCol = 0; ixCol < kernelValues[ixK][ixD][ixRow].Length; ixCol++)
                                {
                                    //Escribir valor por valor
                                    vals += $"{kernelValues[ixK][ixD][ixRow][ixCol]},";
                                }
                            }
                            if (vals.Length > 0)
                                vals = vals.Remove(vals.Length - 1);
                            W.WriteLine(vals);
                        }
                    }
                }
                //Escribir la arquitectura Neuronal (Número de Capas, Cantidad de Neuronas por capa, Función de propagación, función de activación, función de salida, método de error)
                W.WriteLine(FULLY);
                string archNN = ";";
                string[] values = new string[NNModel.NetLayers.Length];
                //Por cada una de las capas
                for (int ixL = 0; ixL < NNModel.NetLayers.Length; ixL++)
                {
                    archNN += NNModel.NetLayers[ixL].Neurons.Length + ";";
                    //if (ixL == 0)
                    //    values[ixL] = $"{NNModel.NetLayers[ixL].WeightedConnections}";
                    //else
                    if (ixL > 0)
                    {
                        values[ixL] = $"{NNModel.NetLayers[ixL].WeightedConnections}_";
                        //Añadir los valores de pesos
                        //Por cada una de las neuronas de la capa
                        for (int ixN = 0; ixN < NNModel.NetLayers[ixL].Neurons.Length; ixN++)
                        {
                            //Por cada una de sus entradas, obtener el peso que tiene actualmente
                            for (int ixIn = 0; ixIn < NNModel.NetLayers[ixL].Neurons[ixN].Inputs.Length; ixIn++)
                            {
                                //Y escribirlo
                                values[ixL] += NNModel.NetLayers[ixL].Neurons[ixN].Inputs[ixIn].Weight + ";";
                            }
                        }
                    }
                }
                archNN = archNN.Remove(archNN.Length - 1);
                W.WriteLine(archNN);
                W.WriteLine($@"//LA CAPA 0 (LA DE ENTRADA) NO SE INCLUYE, POR LO QUE LOS SIGUIENTES VALORES SON DE LA CAPA 1 EN ADELANTE");
                for (int ixV = 1; ixV < values.Length; ixV++)
                    W.WriteLine(values[ixV].Remove(values[ixV].Length - 1));
            }
        }
        /// <summary>
        /// Lee la descripción del modelo RNC contenida dentro de un archivo .txt (arquitectura y pesos) (Aún no totalmente automatizado, ya que se deben considerar como parámetros algunos campos que deberían estar en el archivo)
        /// </summary>
        /// <param name="FilePath">Ruta absoluta del archivo (con el nombre y extensión)</param>
        /// <param name="imgSizeIn">Tamaño de las imágenes en la que el modelo fue entrenado</param>
        /// <param name="imgDimenIn">Dimensión de las imágenes en la que el modelo fue entrenado</param>
        /// <param name="totalClases">Número de clases para las que el modelo fue entrenado</param>
        /// <param name="CNN">La sección convolutiva del modelo leído</param>
        /// <param name="NN">La sección neuronal del modelo leído</param>
        /// <param name="kValues">Vector con los valores de pesos convolutivos del modelo leído</param>
        /// <param name="nValues">Vector con los valores de pesos neuronales del modelo leído</param>
        public static void ReadConvolutionalNeuralNetworkModel(string FilePath, int imgSizeIn, int imgDimenIn, int totalClases, out ConvolutionNetwork CNN, out NeuralNetwork NN, out double[] kValues, out double[] nValues)
        {
            var F = File.OpenRead(FilePath);
            // Read file using StreamReader. Reads file line by line  
            using (StreamReader file = new StreamReader(F))
            {
                string ln;
                List<ConvLayer> convLayers = new List<ConvLayer>();
                List<double> kernelValues = new List<double>();
                List<int> neuralLayers = new List<int>();
                List<double> neuralValues = new List<double>();
                while ((ln = file.ReadLine()) != null)
                {
                    if (ln.Equals(CONVOLUTION))
                    {
                        //Es la sección convolutiva
                        while (ln != null && !ln.Equals(FULLY))
                        {
                            if (ln.StartsWith($"//Capa convolucional"))
                            {
                                //Es el inicio de una capa convolutiva
                                //Obtener los valores
                                ln = file.ReadLine();
                                string[] valsConv = ln.Split(';');
                                int totalK = Convert.ToInt32(valsConv[0]);
                                int sizeK = Convert.ToInt32(valsConv[1]);
                                int strideK = Convert.ToInt32(valsConv[2]);
                                int paddingK = Convert.ToInt32(valsConv[3]);
                                int sizeP = Convert.ToInt32(valsConv[4]);
                                int strideP = Convert.ToInt32(valsConv[5]);
                                //Crear la capa
                                convLayers.Add(new ConvLayer(totalK, sizeK, strideK, paddingK, sizeP, strideP));
                                while ((ln = file.ReadLine()) != null && !ln.StartsWith($"//Capa convolucional") && !ln.Equals(FULLY))
                                {
                                    //Armar una lista con los valores obtenidos
                                    if (!ln.StartsWith(@"//") && ln.Contains(','))
                                    {
                                        string[] vals = ln.Split(',');
                                        for (int ixv = 0; ixv < vals.Length; ixv++)
                                        {
                                            double v = Convert.ToDouble(vals[ixv]);
                                            kernelValues.Add(v);
                                        }
                                    }
                                }
                            }
                            else
                                ln = file.ReadLine();
                        }
                    }
                    if (ln.Equals(FULLY))
                    {
                        ln = file.ReadLine();
                        if (ln.StartsWith(';'))
                            ln = Regex.Replace(ln, @"^;", "");
                        string[] valsNeural = ln.Split(';');
                        for (int ixN = 0; ixN < valsNeural.Length; ixN++)
                        {
                            //Almacenar el número de neuronas por capa
                            neuralLayers.Add(Convert.ToInt32(valsNeural[ixN]));
                        }
                        //Cada línea a partir de la siguiente que se lea son los valores aprendidos de los pesos por cada una de las capas
                        while ((ln = file.ReadLine()) != null)
                        {
                            if (!ln.StartsWith(@"//"))
                            {
                                ln = ln.Substring(ln.IndexOf('_') + 1);
                                string[] valsn = ln.Split(';');
                                for (int ixvn = 0; ixvn < valsn.Length; ixvn++)
                                {
                                    neuralValues.Add(Convert.ToDouble(valsn[ixvn]));
                                }
                            }
                        }
                    }
                }
                file.Close();
                ConvLayer[] l = convLayers.ToArray();
                CNN = new ConvolutionNetwork();
                int iso, ido;
                int totalKernelValues = CNN.Build(l, imgSizeIn, imgDimenIn, out iso, out ido);
                NN = new NeuralNetwork();
                int[] NNLayers = neuralLayers.ToArray(); //{ (iso * iso * ido), 100, 70, totalClases };
                NeuralNetwork.ErrorMethod errorMethod = NeuralNetwork.ErrorMethod.MeanSquaredError;
                int totalWeigthsValues = NN.Build(NNLayers, PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal,
                    PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal, errorMethod);
                kValues = kernelValues.ToArray();
                nValues = neuralValues.ToArray();
            }
        }
        #endregion
        /// <summary>
        /// Clase dedicada a leer los archivos MNIST
        /// Adaptada a partir de la publicada por el usuario "koryakinp" de StackOverflow
        /// Source: https://stackoverflow.com/questions/49407772/reading-mnist-database
        /// </summary>
        public static class MnistReader
        {
            public static IEnumerable<ImageML> ReadTrainData(string TrainImages, string TrainLabels)
            {
                foreach (var item in Read(TrainImages, TrainLabels))
                {
                    yield return item;
                }
            }
            public static IEnumerable<ImageML> ReadTestData(string TestImages, string TestLabels)
            {
                foreach (var item in Read(TestImages, TestLabels))
                {
                    yield return item;
                }
            }
            public static Data[] ReadData(string imgPath, string lblPath, string[] categories, int maxImages)
            {
                var Images = Read(imgPath, lblPath);
                Data[] D;
                if (maxImages <= 0)
                    D = new Data[Images.Count()];
                else
                    D = new Data[maxImages * categories.Length];
                int ixD = 0;
                int max = 0;
                int[] imagesPerCat = new int[categories.Length];
                for (int ixCat = 0; ixCat < imagesPerCat.Length; ixCat++)
                {
                    imagesPerCat[ixCat] = maxImages;
                }
                foreach (ImageML item in Images)
                {
                    imagesPerCat[item.Label]--;
                    if (imagesPerCat[item.Label] >= 0)
                    {
                        //Realizar el parseo
                        double[] values = new double[item.Pixels.Length];
                        int[] expected = new int[categories.Length];
                        //Inicializar todos los valores de 'expected' en -1 excepto el que debe ser
                        for (int ixE = 0; ixE < categories.Length; ixE++)
                            expected[ixE] = -1;
                        expected[item.Label] = 1;
                        for (int ixH = 0; ixH < item.Pixels.Length; ixH++)
                        {
                            if (item.Pixels[ixH] > 0)
                                //values[ixH] = Convert.ToDouble(item.Pixels[ixH]) / 255.0;
                                values[ixH] = item.Pixels[ixH];
                        }
                        //Al ser MNIST, sólo tiene 1 dimensión, ponerla estática
                        double[][][] valJagg = GetJaggedArray_FromVector(values, 1);
                        D[ixD] = new Data(values, valJagg, expected, categories);
                        ixD++;
                        max++;
                        if (imagesPerCat.Where(i => i > 0).Count() == 0)
                            break;
                    }
                }
                return D;
            }
            public static Data[] ReadData_TestOneCat(string imgPath, string lblPath, int[] cats4Read, int maxImages)
            {
                var Images = Read(imgPath, lblPath);
                Data[] D;
                if (maxImages <= 0)
                    D = new Data[Images.Count()];
                else
                    D = new Data[maxImages * cats4Read.Length];
                int ixD = 0;
                int max = 0;
                int[] imagesPerCat = new int[cats4Read.Length];
                for (int ixCat = 0; ixCat < imagesPerCat.Length; ixCat++)
                {
                    imagesPerCat[ixCat] = maxImages;
                }
                string[] categories = { "SÍ LO ES", "NO LO ES" };
                foreach (ImageML item in Images)
                {
                    if (cats4Read.Contains(item.Label))
                    {
                        //Obtener el index del número
                        int index = Array.IndexOf(cats4Read, item.Label);
                        imagesPerCat[index]--;
                        if (imagesPerCat[index] >= 0)
                        {
                            //Realizar el parseo
                            double[] values = new double[item.Pixels.Length];
                            int[] expected = new int[categories.Length];
                            if (index == 0)
                            {
                                //Se trata de la clasificación original, por lo que el expected debe ser 1,-1
                                expected[0] = 1;
                                expected[1] = -1;
                            }
                            else
                            {
                                //Se trata de las demás clasificaciones, por lo que el expected debe ser -1,1
                                expected[0] = -1;
                                expected[1] = 1;
                            }
                            for (int ixH = 0; ixH < item.Pixels.Length; ixH++)
                            {
                                if (item.Pixels[ixH] > 0)
                                    //values[ixH] = Convert.ToDouble(item.Pixels[ixH]) / 255.0;
                                    values[ixH] = item.Pixels[ixH];
                            }
                            //Al ser MNIST, sólo tiene 1 dimensión, ponerla estática
                            double[][][] valJagg = GetJaggedArray_FromVector(values, 1);
                            D[ixD] = new Data(values, valJagg, expected, categories);
                            ixD++;
                            max++;
                            if (imagesPerCat.Where(i => i > 0).Count() == 0)
                                break;
                        }
                    }
                }
                return D;
            }
            public static Data[] ReadData_TrainOneCat(string imgPath, string lblPath, int cat4Read, int maxImages)
            {
                var Images = Read(imgPath, lblPath);
                Data[] D;
                if (maxImages <= 0)
                    D = new Data[Images.Count()];
                else
                    D = new Data[maxImages * 2];
                int ixD = 0;
                int max = 0;
                int imagesPerCat = maxImages;
                int differentImages = maxImages;
                string[] categories = { "SÍ LO ES", "NO LO ES" };
                foreach (ImageML item in Images)
                {
                    if (item.Label == cat4Read)
                    {
                        //Colocar imágenes de la clase que se busca
                        imagesPerCat--;
                        if (imagesPerCat >= 0)
                        {
                            //Realizar el parseo
                            double[] values = new double[item.Pixels.Length];
                            int[] expected = { 1, -1 };
                            for (int ixH = 0; ixH < item.Pixels.Length; ixH++)
                            {
                                if (item.Pixels[ixH] > 0)
                                    //values[ixH] = Convert.ToDouble(item.Pixels[ixH]) / 255.0;
                                    values[ixH] = item.Pixels[ixH];
                            }
                            //Al ser MNIST, sólo tiene 1 dimensión, ponerla estática
                            double[][][] valJagg = GetJaggedArray_FromVector(values, 1);
                            D[ixD] = new Data(values, valJagg, expected, categories);
                            ixD++;
                            max++;
                        }
                    }
                    //Colocar imágenes "basura" (que no son la clase esperada)
                    else
                    {
                        differentImages--;
                        if (differentImages >= 0)
                        {
                            //Realizar el parseo
                            double[] values = new double[item.Pixels.Length];
                            int[] expected = { -1, 1 };
                            for (int ixH = 0; ixH < item.Pixels.Length; ixH++)
                            {
                                if (item.Pixels[ixH] > 0)
                                    //values[ixH] = Convert.ToDouble(item.Pixels[ixH]) / 255.0;
                                    values[ixH] = item.Pixels[ixH];
                            }
                            //Al ser MNIST, sólo tiene 1 dimensión, ponerla estática
                            double[][][] valJagg = GetJaggedArray_FromVector(values, 1);
                            D[ixD] = new Data(values, valJagg, expected, categories);
                            ixD++;
                            max++;
                        }
                    }
                    if (imagesPerCat == 0 && differentImages == 0)
                    {
                        break;
                    }
                }
                return D;
            }
            private static IEnumerable<ImageML> Read(string imagesPath, string labelsPath)
            {
                BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
                BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

                //0000     32 bit integer  0x00000803(2051) magic number
                int magicNumber = images.ReadBigInt32();
                //0004     32 bit integer  60000            number of images
                int numberOfImages = images.ReadBigInt32();
                //0008     32 bit integer  28               number of rows
                int width = images.ReadBigInt32();
                //0012     32 bit integer  28               number of columns
                int height = images.ReadBigInt32();

                //0000     32 bit integer  0x00000801(2049) magic number (MSB first)
                int magicLabel = labels.ReadBigInt32();
                //0004     32 bit integer  60000            number of items
                int numberOfLabels = labels.ReadBigInt32();

                for (int i = 0; i < numberOfImages; i++)
                {
                    var bytes = images.ReadBytes(width * height);
                    var arr = new byte[height, width];
                    arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                    yield return new ImageML()
                    {
                        Data = arr,
                        Label = labels.ReadByte(),
                        Pixels = bytes
                    };
                }
                labels.Close();
                images.Close();
            }
        }
    }
    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
    public class ImageML
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }
        public byte[] Pixels { get; set; }
    }
}
