using CNN_RySI.CNN;
using CNN_RySI.CNN.Components;
using CNN_RySI.MLP;
using CNN_RySI.MLP.Functions;
using CNN_RySI.Structures;
using CNN_RySI.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN_RySI.Train
{
    class Trainers
    {

        /// <summary>
        /// T-I. Método que crea n cantidad de individuos de Arquitectura de Redes Neuronales Convolutivas para entrenarlas en 'paralelo'
        /// </summary>
        /// <param name="TrainDataset">Conjunto de imágenes de entrenamiento</param>
        /// <param name="TestDataSet">Conjunto de imágenes de prueba</param>
        /// <param name="TotalCategories">Número de categorías en las cuales se entrenarán los modelos neuronales</param>
        /// <returns></returns>
        public static EvoAIndividual[] Hyperheuristic_Trainning(Data[] TrainDataset, Data[] TestDataSet, int TotalCategories)
        {
            //string Results = "";
            //Se almacenarán los 4 primeros lugares
            EvoAIndividual[] Results = new EvoAIndividual[4];
            //Obtener la primera imagen del dataset para tener cantidades de tamaño. Se presupone que todas las imágenes comparten las mismas dimensiones.
            double[][][] Image = TrainDataset[0].Values_Jagged;
            //Obtener la dimensión de las imágenes
            int imgDimensionIn_DataSet = Image.Length;
            //Obtener el tamaño (ancho) de las imágenes (se presuponen que son cuadradas)
            int imgSizeIn_DataSet = Image[0].Length;
            int imgSizeOut = 0;
            int imgDimensionOut = 0;
            #region DECLARACIÓN DE HIPERPARÁMETROS **************************************
            //PARA LAS ARQUITECTURAS (SON FIJAS)
            const int MinConvLayers = 1;
            const int MaxConvLayers = 3;
            const int MinKernelsPerLayer = 2;
            const int MaxKernelsPerLayer = 24;
            const int MinKernelSize = 3;
            const int MaxKernelSize = 7;
            const int MinKernelStride = 1;
            const int MaxKernelStride = 3;
            const int MinConvPadding = 0;
            const int MaxConvPadding = 4;
            const int PoolingSize = 2;
            const int PoolingStride = 2;
            const int MinNeuralHiddenLayers = 1;
            const int MaxNeuralHiddenLayers = 4;
            const int MinNeuronsPerLayer = 50;
            const int MaxNeuronsPerLayer = 200;
            const double MinErrorTolerance = 0.05;
            const double MaxErrorTolerance = 0.20;
            //La imagen obtenida del proceso convolutivo no debe ser menor que 4 ni mayor a la imagen original
            const int MinImgSizeExpected = 4;
            const int MinEpochs = 2;
            const int MaxEpochs = 5;
            const int MinBatchSize = 5;
            const int MaxBatchSize = 40;
            int MaxImgSizeExpected = imgSizeIn_DataSet;
            //Para la programación evolutiva de los individuos de la arquitectura
            const int TotalInitialAPop = 5;
            const int MaxGenerationsArch = 5;
            //const double MaxMinutesPerArchGen = 5760;       //5760 minutos, 96 horas, 4 días máximo por generación 
            const double MaxMinutesPerArchGen = 10;       //60 minutos, 1 hora máximo por generación (5 + 1 = 6 horas aprox la ejecución total)
            int TotalAPop = TotalInitialAPop * 2;
            //string HiperParametersArch = $"";
            int TotalComp = Convert.ToInt32(TotalAPop * 0.66);    //Total de competiciones estocásticas que se aplicarán
            #endregion
            Console.WriteLine($"CREACIÓN DE LA POBLACIÓN INICIAL DE ARQUITECTURAS... - {DateTime.Now}");
            #region CREACIÓN DE LA POBLACIÓN INICIAL ************************************
            EvoAIndividual[] ArrayITotalAPop = new EvoAIndividual[TotalAPop];
            //EvoAIndividual[] BestArchitectures = new EvoAIndividual[3]; //Ranking de las 3 mejores arquitecturas
            //Para cada uno de los individuos
            for (int ixArch = 0; ixArch < TotalInitialAPop; ixArch++)
            {
                //Obtener los parámetros necesarios de forma aleatoria para la creación del individuo
                //Configuración Convolutiva
                bool ValidConvConfig = false;
                ConvLayer[] convLayers = null;
                while (!ValidConvConfig)
                {
                    int totalConvLayers = RandomHelper.GetRandomIntAbsolute(MinConvLayers, MaxConvLayers);    //Obtener randomly
                    //Capas convolutivas
                    convLayers = new ConvLayer[totalConvLayers];
                    for (int ixCL = 0; ixCL < totalConvLayers; ixCL++)
                    {
                        int totalKernels = RandomHelper.GetRandomIntAbsolute(MinKernelsPerLayer, MaxKernelsPerLayer);   //Obtener randomly
                        int kSize = RandomHelper.GetRandomIntAbsolute(MinKernelSize, MaxKernelSize);                    //Obtener randomly
                        int kStride = RandomHelper.GetRandomIntAbsolute(MinKernelStride, MaxKernelStride);              //Obtener randomly
                        int kPadding = RandomHelper.GetRandomIntAbsolute(MinConvPadding, MaxConvPadding);               //Obtener randomly
                        bool hasPadding = RandomHelper.GetRandomBoolCoin();                                             //Obtener randomly
                        if (hasPadding)
                            convLayers[ixCL] = new ConvLayer(totalKernels, kSize, kStride, kPadding, PoolingSize, PoolingStride);
                        else
                            convLayers[ixCL] = new ConvLayer(totalKernels, kSize, kStride, kPadding);
                    }
                    //Tratar de construir un modelo Dummy, para validar si las configuraciones obtenidas vía aleatoria son válidas para producir una red Convolutiva
                    ConvolutionNetwork ConvDummyModel = new ConvolutionNetwork();
                    try
                    {
                        //Ejecutar el método de construcción, si da error, mandará un throw error que se controlará, reiniciando la configuración Convolutiva
                        ConvDummyModel.Build(convLayers, imgSizeIn_DataSet, imgDimensionIn_DataSet, out imgSizeOut, out imgDimensionOut);
                        //En caso de que se haya podido construir, hacerlo válido sólo si el tamaño de la imagen está dentro de los límites
                        ValidConvConfig = imgSizeOut >= MinImgSizeExpected && imgSizeOut <= MaxImgSizeExpected;
                    }
                    catch (Exception) { ValidConvConfig = false; }
                }
                //Configuración Neuronal
                //Obtener el número total de capas, más 2 (la de entrada y la de salida)
                int totalNeuralLayers = RandomHelper.GetRandomIntAbsolute(MinNeuralHiddenLayers, MaxNeuralHiddenLayers) + 2;      //Obtener randomly
                //Crear el vector de capas neuronales
                int[] neuralLayers = new int[totalNeuralLayers];
                neuralLayers[0] = imgSizeOut * imgSizeOut * imgDimensionOut;
                for (int ixN = 1; ixN < totalNeuralLayers - 1; ixN++)
                {
                    int totalNeurons = RandomHelper.GetRandomIntAbsolute(MinNeuronsPerLayer, MaxNeuronsPerLayer);       //Obtener randomly
                    neuralLayers[ixN] = totalNeurons;
                }
                neuralLayers[totalNeuralLayers - 1] = TotalCategories;
                //Demás variables de arquitectura
                int epochs = RandomHelper.GetRandomIntAbsolute(MinEpochs, MaxEpochs);         //Obtener randomly
                int batchSize = RandomHelper.GetRandomIntAbsolute(MinBatchSize, MaxBatchSize);      //Obtener randomly
                double errorTolerance = RandomHelper.GetRandomDouble(MinErrorTolerance, MaxErrorTolerance);  //Obtener randomly
                //Crear el individuo que representará una configuración de arquitectura
                ArrayITotalAPop[ixArch] = new EvoAIndividual(ixArch, convLayers, neuralLayers, epochs, batchSize, errorTolerance, -1);
            }
            #endregion
            #region REALIZAR EL PROCESO DE ENTRENAMIENTO EVOLUTIVO PARA CADA UNO DE LOS INDIVIDUOS *************************
            Console.WriteLine($"CALCULANDO LA APTITUD DE LA POBLACIÓN INICIAL... (MÁX MIN ESPERA : {MaxMinutesPerArchGen}) - {DateTime.Now}");
            #region CALCULAR LA APTITUD DE LA POBLACIÓN INICIAL
            Task<EvoAIndividual>[] ProcessInitialPop = new Task<EvoAIndividual>[TotalInitialAPop];
            //Crear la instancia de tiempo
            DateTime DeadLine = DateTime.Now.AddMinutes(MaxMinutesPerArchGen);
            //Para cada una de las arquitecturas
            for (int ixArch = 0; ixArch < TotalInitialAPop; ixArch++)
            {
                ProcessInitialPop[ixArch] = ProcessEAI(ArrayITotalAPop[ixArch], imgSizeIn_DataSet, imgDimensionIn_DataSet, TrainDataset, TestDataSet, DeadLine);
            }
            //Esperar al resultado de todos los procesos
            Task<EvoAIndividual[]> CombinedTasksInitialPop = Task.WhenAll(ProcessInitialPop);
            EvoAIndividual[] ResultsInitialPop = CombinedTasksInitialPop.Result;
            //Por cada uno de los resultados, actualizarlos en el array de individuos
            for (int ixR = 0; ixR < ResultsInitialPop.Length; ixR++)
            {
                int ixInP = ResultsInitialPop[ixR].ID;
                ArrayITotalAPop[ixInP] = ResultsInitialPop[ixR];
            }
            #endregion
            int ActualGenArch = 0;
            double BestAccuracyOfAll = 0;
            while (ActualGenArch < MaxGenerationsArch || BestAccuracyOfAll < 0.99)
            {
                Console.WriteLine($"[{ActualGenArch}°] GENERACIÓN DE ARQUITECTURAS... - {DateTime.Now}");
                Console.WriteLine($"CREACIÓN DE LOS INDIVIDUOS 'MUTADOS' - {DateTime.Now}");
                #region REALIZAR LA "MUTACIÓN" POR REEMPLAZO
                //Para la mutación se considera el reemplazo total del individuo por uno nuevo, por lo que se vuelven a crear
                for (int ixArch = 0; ixArch < TotalInitialAPop; ixArch++)
                {
                    //Obtener los parámetros necesarios de forma aleatoria
                    bool ValidConvConfig = false;
                    ConvLayer[] convLayers = null;
                    while (!ValidConvConfig)
                    {
                        int totalConvLayers = RandomHelper.GetRandomIntAbsolute(MinConvLayers, MaxConvLayers);
                        //Capas convolutivas
                        convLayers = new ConvLayer[totalConvLayers];
                        for (int ixCL = 0; ixCL < totalConvLayers; ixCL++)
                        {
                            int totalKernels = RandomHelper.GetRandomIntAbsolute(MinKernelsPerLayer, MaxKernelsPerLayer);
                            int kSize = RandomHelper.GetRandomIntAbsolute(MinKernelSize, MaxKernelSize);
                            int kStride = RandomHelper.GetRandomIntAbsolute(MinKernelStride, MaxKernelStride);
                            int kPadding = RandomHelper.GetRandomIntAbsolute(MinConvPadding, MaxConvPadding);
                            bool hasPadding = RandomHelper.GetRandomBoolCoin();
                            if (hasPadding)
                                convLayers[ixCL] = new ConvLayer(totalKernels, kSize, kStride, kPadding, PoolingSize, PoolingStride);
                            else
                                convLayers[ixCL] = new ConvLayer(totalKernels, kSize, kStride, kPadding);
                        }
                        //Tratar de construir un modelo Dummy, validándolo
                        ConvolutionNetwork ConvDummyModel = new ConvolutionNetwork();
                        try
                        {
                            ConvDummyModel.Build(convLayers, imgSizeIn_DataSet, imgDimensionIn_DataSet, out imgSizeOut, out imgDimensionOut);
                            //En caso de que se haya podido construir, hacerlo válido sólo si el tamaño de la imagen está dentro de los límites
                            ValidConvConfig = imgSizeOut >= MinImgSizeExpected && imgSizeOut <= MaxImgSizeExpected;
                        }
                        catch (Exception) { ValidConvConfig = false; }
                    }
                    //Capas neuronales
                    int totalNeuralLayers = RandomHelper.GetRandomIntAbsolute(MinNeuralHiddenLayers, MaxNeuralHiddenLayers) + 2;
                    int[] neuralLayers = new int[totalNeuralLayers];
                    neuralLayers[0] = imgSizeOut * imgSizeOut * imgDimensionOut;
                    for (int ixN = 1; ixN < totalNeuralLayers - 1; ixN++)
                    {
                        int totalNeurons = RandomHelper.GetRandomIntAbsolute(MinNeuronsPerLayer, MaxNeuronsPerLayer);
                        neuralLayers[ixN] = totalNeurons;
                    }
                    neuralLayers[totalNeuralLayers - 1] = TotalCategories;
                    //Demás variables de arquitectura
                    int epochs = RandomHelper.GetRandomIntAbsolute(MinEpochs, MaxEpochs);
                    int batchSize = RandomHelper.GetRandomIntAbsolute(MinBatchSize, MaxBatchSize);
                    double errorTolerance = RandomHelper.GetRandomDouble(MinErrorTolerance, MaxErrorTolerance);
                    //Crear el individuo que representará una configuración de arquitectura ("Mutado")
                    ArrayITotalAPop[ixArch + TotalInitialAPop] = new EvoAIndividual(ixArch + TotalInitialAPop, convLayers, neuralLayers, epochs, batchSize, errorTolerance, ActualGenArch);
                    //Reiniciar el contador de victorias del individuo original
                    ArrayITotalAPop[ixArch].Victories = 0;
                }
                #endregion
                Console.WriteLine($"CALCULANDO LA APTITUD DE LA POBLACIÓN MUTADA... (MÁX MIN ESPERA: {MaxMinutesPerArchGen}) - {DateTime.Now}");
                #region CÁLCULO DE LA APTITUD DE LA POBLACIÓN OBTENIDA
                //Realizar el cálculo de la aptitud
                Task<EvoAIndividual>[] ProcessInitialMutants = new Task<EvoAIndividual>[TotalInitialAPop];
                DeadLine = DateTime.Now.AddMinutes(MaxMinutesPerArchGen);
                //Para cada una de las arquitecturas
                for (int ixArch = 0; ixArch < TotalInitialAPop; ixArch++)
                {
                    ProcessInitialMutants[ixArch] = ProcessEAI(ArrayITotalAPop[ixArch + TotalInitialAPop], imgSizeIn_DataSet, imgDimensionIn_DataSet, TrainDataset, TestDataSet, DeadLine);
                }
                //Esperar al resultado de todos los procesos
                Task<EvoAIndividual[]> CombinedTasksMutants = Task.WhenAll(ProcessInitialMutants);
                EvoAIndividual[] ResultsMutants = CombinedTasksMutants.Result;
                //Por cada uno de los resultados, actualizarlos en el array de individuos
                for (int ixR = 0; ixR < ResultsMutants.Length; ixR++)
                {
                    int ixInP = ResultsMutants[ixR].ID;
                    ArrayITotalAPop[ixInP] = ResultsMutants[ixR];
                }
                #endregion
                Console.WriteLine($"REALIZANDO LA COMPETICIÓN ESTOCÁSTICA... - {DateTime.Now}");
                #region REALIZAR LA COMPETICIÓN ESTOCÁSTICA
                //Para la competición estocástica, se tomarán en cuenta únicamente el porcentaje de precisión obtenido. Mientras más, mejor.
                //REALIZAR EL TORNEO ESTOCÁSTICO ENTRE TODA LA POBLACIÓN
                for (int ixWT = 0; ixWT < TotalAPop; ixWT++)
                {
                    Random r = new Random();
                    //...Realizar n cantidad de competiciones aleatorias entre miembros de la población total
                    for (int t = 0; t < TotalComp; t++)
                    {
                        int IxComp;
                        //Obtener un index de competición (IxComp) aleatorio que sea diferente del index de pesos actual (ixW)
                        do
                            IxComp = r.Next(0, TotalAPop - 1);
                        while (ixWT == IxComp);
                        if (ArrayITotalAPop[ixWT].AccuracyPercent > ArrayITotalAPop[IxComp].AccuracyPercent)
                            ArrayITotalAPop[ixWT].Victories++;
                    }
                }
                #endregion
                Console.WriteLine($"SELECCIÓN DE LA MITAD MÁS APTA... - {DateTime.Now}");
                #region SELECCIÓN DE LOS MÁS APTOS
                //ORDENAR LA POBLACIÓN DE PESOS CONFORME A LAS VICTORIAS
                ArrayITotalAPop = ArrayITotalAPop.OrderByDescending(w => w.Victories).ToArray();
                //LA MITAD MÁS ALTA SERÁ LA NUEVA POBLACIÓN INICIAL
                string tempData = "";
                for (int ixE = 0; ixE < ArrayITotalAPop.Length; ixE++)
                {
                    ArrayITotalAPop[ixE].ID = ixE;
                    tempData += $"[{ixE}] {ArrayITotalAPop[ixE].AccuracyPercent}%, {ArrayITotalAPop[ixE].Victories} Victories|";
                }
                Console.WriteLine(tempData);
                if (ArrayITotalAPop[0].AccuracyPercent > BestAccuracyOfAll)
                    BestAccuracyOfAll = ArrayITotalAPop[0].AccuracyPercent;
                #endregion
                ActualGenArch++;
            }
            Console.WriteLine($"SELECCIÓN DE LOS FINALISTAS... - {DateTime.Now}");
            //Una vez realizado todo el proceso, seleccionar los 3 mejores individuos y mostrarlos
            //Results += $"1ª => {GetArchIndividualData(ArrayITotalAPop[0])} |";
            //Results += $"2ª => {GetArchIndividualData(ArrayITotalAPop[1])} |";
            //Results += $"3ª => {GetArchIndividualData(ArrayITotalAPop[2])} |";
            Results[0] = ArrayITotalAPop[0];
            Results[1] = ArrayITotalAPop[1];
            Results[2] = ArrayITotalAPop[2];
            Results[3] = ArrayITotalAPop[3];
            #endregion
            return Results;
        }
        /// <summary>
        /// T-II. Método que crea una RNC con los datos de un individuo de Arquitectura, entrena la RNC y predice dando un porcentaje de acierto, todo dentro de un Task (Hilo)
        /// </summary>
        /// <param name="EAI">Individuo de Arquitectura a ser procesado</param>
        /// <param name="imgSizeIn_DataSet">Tamaño estándar de las imágenes de entrada (ancho o alto)</param>
        /// <param name="imgDimensionIn_DataSet">Dimensión estándar de las imágenes de entrada (canales RGB o Grises)</param>
        /// <param name="TrainDataSet">Conjunto de imágenes de entrenamiento</param>
        /// <param name="TestDataSet">Conjunto de imágenes de prueba</param>
        /// <param name="DeadLine">Tiempo límite (en fecha) para el entrenamiento</param>
        /// <returns></returns>
        private static Task<EvoAIndividual> ProcessEAI(EvoAIndividual EAI,
            int imgSizeIn_DataSet, int imgDimensionIn_DataSet, Data[] TrainDataSet, Data[] TestDataSet, DateTime DeadLine)
        {
            return Task.Factory.StartNew(() =>
            {
                int imgSizeOut, imgDimensionOut;
                //Crear los modelos
                ConvolutionNetwork CNN = new ConvolutionNetwork();
                int totalValsKernel = CNN.Build(EAI.CLayers, imgSizeIn_DataSet, imgDimensionIn_DataSet, out imgSizeOut, out imgDimensionOut);
                NeuralNetwork NN = new NeuralNetwork();
                int totalValsWeights = NN.Build(EAI.NLayers, PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal,
                    PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal, NeuralNetwork.ErrorMethod.MeanSquaredError);
                //Realizar el entrenamiento del individuo (Al ser el entrenamiento en paralelo, se crearán nuevos hilos o Tasks a partir de este)
                Metaheuristic_TrainWeights_Offline_Parallel_OneCategory(totalValsKernel, totalValsWeights, DeadLine, TrainDataSet, imgSizeIn_DataSet, imgDimensionIn_DataSet,
                    EAI, CNN, NN, out CNN, out NN, out double TotalMinutes, out string preH, out string[] summary);
                //Realizar el testing del producto de cada una de las arquitecturas
                int hits = Hyperheuristic_Testing(TestDataSet, CNN, NN);
                EAI.TotalMinutesRequired = TotalMinutes;
                EAI.AccuracyPercent = (hits / Convert.ToDouble(TestDataSet.Length)) * 100;
                EAI.PreHyperparameters = preH;
                EAI.TrainingSummary = summary;
                EAI.TrainedCNN = CNN;
                EAI.TrainedNN = NN;
                return EAI;
            });
        }
        /// <summary>
        /// T-III. Método que realiza el entrenamiento evolutivo de una RNC auxiliándose de un selector Metaheurístico
        /// </summary>
        /// <param name="totalValKernel">Longitud del vector de pesos de Kernel (convolutivos)</param>
        /// <param name="totalValWeight">Longitud del vector de pesos sinápticos (neuronales)</param>
        /// <param name="DeadLine">Tiempo límite (en fecha) para el entrenamiento</param>
        /// <param name="DataSet">Conjunto de imágenes de entrenamiento</param>
        /// <param name="imgSizeIn">Tamaño estándar de las imágenes de entrada (ancho o alto)</param>
        /// <param name="imgDimensionIn">Dimensión estándar de las imágenes de entrada (canales RGB o Grises)</param>
        /// <param name="EAI">Individuo de Arquitectura a ser procesado</param>
        /// <param name="CNNModel">Sección de la RNC (Parte convolutiva) ya construida</param>
        /// <param name="NNModel">Sección de la RNC (Parte neuronal) ya construida</param>
        /// <param name="TrainnedCNNModel">Sección de la RNC (Parte convolutiva) ya entrenada</param>
        /// <param name="TrainnedNNModel">Sección de la RNC (Parte neuronal) ya entrenada</param>
        /// <param name="TotalMinutesRequired">Total de minutos que requirió el entrenamiento (NO 100% PRECISO, AL USAR HILOS)</param>
        /// <param name="PreHyperParams">Detalle de los hiperparámetros</param>
        /// <param name="ResumeMetaSelector">Detalle del entrenamiento</param>
        public static void Metaheuristic_TrainWeights_Offline_Parallel_OneCategory(int totalValKernel, int totalValWeight, DateTime DeadLine, 
            Data[] DataSet, int imgSizeIn, int imgDimensionIn, EvoAIndividual EAI, ConvolutionNetwork CNNModel, NeuralNetwork NNModel, 
            out ConvolutionNetwork TrainnedCNNModel, out NeuralNetwork TrainnedNNModel, out double TotalMinutesRequired, 
            out string PreHyperParams, out string[] ResumeMetaSelector)
        {
            #region Declaración de Hiperparámetros
            int totalValues4Training = totalValKernel + totalValWeight;
            const int InitialPop = 14;      //Población inicial de vectores de pesos (que contendrán los valores de los pesos de kernel y los de las conexiones neuronales)
            const double MinKVal = -1;      //Valor mínimo de los pesos de los Kernels
            const double MaxKVal = 2;       //Valor máximo de los pesos de los Kernels
            const int DecimalsKVal = 1;     //Número máximo de decimales que tendrán los pesos de los Kernels
            const double MinWVal = -4;      //Valor mínimo de los pesos de las conexiones neuronales
            const double MaxWVal = 4;       //Valor máximo de los pesos de las conexiones neuronales
            const int DecimalsWVal = 5;     //Número máximo de decimales de los pesos de las conexiones neuronales
            const double MinFVal = 0.1;     //Valor mínimo del factor de multiplicación para la evolución diferencial 'dinámica'
            const double MaxFVal = 1.8;     //Valor máximo del factor de multiplicación para la evolución diferencial 'dinámica'
            const int DecimalsFVal = 1;     //Número de decimales del factor de multiplicación para la evolución diferencial 'dinámica'
            const double CompetitionRatio = .66; //Factor de competencias estocásticas a realizar (totalPop / competitionRatio), ej. 2 = 50% de la población total
            const double MaxGens = 4000;   //Número máximo de generaciones que tendrá el entrenamiento evolutivo

            int BatchSize = EAI.BatchSize;      //Número de imágenes que contendrá el subconjunto
            int TotalEpoch = EAI.Epochs;       //Número de épocas que tendrá el entrenamiento offline
            double ErrorLimit = EAI.ErrorTolerance;

            int TotalPop = InitialPop * 2;      //Total de poblaciones del entrenamiento evolutivo
            int TotalComp = Convert.ToInt32(TotalPop * CompetitionRatio);    //Total de competiciones estocásticas que se aplicarán
            int EachGensNewVals = Convert.ToInt32(MaxGens * 0.075); //Número que indicará cada cada cuántas generaciones se reemplazarán n cantidad de la población inicial con valores totalmente nuevos (ej. 0.015 en 10,000 gens, serían cada 150 gens)
            int SizeNewVals = Convert.ToInt32(Math.Floor(InitialPop * 0.9));    //Cantidad de individuos que serán reemplazados (Ej. 0.9, 90%). La cantidad resultante es redondeada hacia abajo (ej. 14 * 0.9 = 12.6 = 12) 
            PreHyperParams = $"InitialPop:{InitialPop}|MinKVal:{MinKVal}|MaxKVal:{MaxKVal}|DecimalsKVal:{DecimalsKVal}|MinWVal:{MinWVal}|MaxWVal:{MaxWVal}|DecimalsKVal:{DecimalsWVal}|" +
                $"MinFVal:{MinFVal}|MaxFVal:{MaxFVal}|DecimalsFVal:{DecimalsFVal}|CompetitionRatio:{CompetitionRatio}|MaxGensPerBatch:{MaxGens}|BatchSize:{BatchSize}|TotalEpochs:{TotalEpoch}|ErrorLimit:{ErrorLimit}|" +
                $"EachGensNewVals:{EachGensNewVals}|SizeNewVals:{SizeNewVals}";
            #endregion
            #region Creación de la población inicial
            //I. Generar una población inicial de vectores de pesos----------------------------
            //Vector con los individuos de la población inicial
            EvoWIndividual[] EvoIArray = new EvoWIndividual[TotalPop];
            //Para obtener el mejor al finalizar el entrenamiento
            EvoWIndividual BestIndividual = new EvoWIndividual();
            //Vector de vectores de pesos (Para la mutación por evolución diferencial)
            double[][] InitialPopW = new double[InitialPop][];
            double[][] InitialPopK = new double[InitialPop][];
            //Generación de la población inicial de pesos
            for (int i = 0; i < InitialPop; i++)
            {
                EvoIArray[i] = new EvoWIndividual(); //suma de extensiones de Kernel mas Pesos
                //Pesos para kernels
                EvoIArray[i].KValues = new double[totalValKernel];
                for (int ix = 0; ix < totalValKernel; ix++)
                    EvoIArray[i].KValues[ix] = RandomHelper.GetRandomDouble(MinKVal, MaxKVal, DecimalsKVal);
                InitialPopK[i] = EvoIArray[i].KValues;
                //Pesos para conexiones neuronales (NN)
                EvoIArray[i].WValues = new double[totalValWeight];
                for (int ix = 0; ix < totalValWeight; ix++)
                    EvoIArray[i].WValues[ix] = RandomHelper.GetRandomDouble(MinWVal, MaxWVal, DecimalsWVal);
                InitialPopW[i] = EvoIArray[i].WValues;
                //Asignación de identificador (para el procesamiento en paralelo) No lo conservará toda su vida, variará cada generación
                EvoIArray[i].ID = i;
            }
            Random r = new Random();
            //ALGORITMO DE ENTRENAMIENTO--------------------------
            //Construcción de la lista de Metaheurísticas
            //7 Funciones de heurísticas de bajo nivel
            EvoFunction[] FunctionStatusK = new EvoFunction[7];
            for (int ixFun = 0; ixFun < FunctionStatusK.Length; ixFun++)
                FunctionStatusK[ixFun] = new EvoFunction();
            //7 Funciones de heurísticas de bajo nivel
            EvoFunction[] FunctionStatusW = new EvoFunction[7];
            for (int ixFun = 0; ixFun < FunctionStatusW.Length; ixFun++)
                FunctionStatusW[ixFun] = new EvoFunction();
            #endregion
            //ESTABLECER UNA BANDERA DE SALIDA DE "EMERGENCIA"
            bool DeadLineReached = false;
            int ixEpoch = 0;
            var timerAllProcess = System.Diagnostics.Stopwatch.StartNew();
            int TotalBatches = 0;
            int NoHaltByGenerationLimit = 0;
            int TotalGensPerEpoch = 0;
            //ResumeMetaSelector = new string[Convert.ToInt32(TotalEpoch * ((DataSet.Length/BatchSize) + 1) * MaxGens)];
            ResumeMetaSelector = new string[TotalEpoch];
            //POR CADA UNO DE LOS EPOCH
            Console.WriteLine($"INICIO DEL ENTRENAMIENTO DE ARQUITECTURA #[ {EAI.ID} ] - {DateTime.Now}");
            while (ixEpoch < TotalEpoch && !DeadLineReached)
            //for (int ixEpoch = 0; ixEpoch < TotalEpoch; ixEpoch++)
            {
                var timerEpoch = System.Diagnostics.Stopwatch.StartNew();
                NoHaltByGenerationLimit = 0;
                TotalBatches = 0;
                TotalGensPerEpoch = 0;
                int ixA = 0;
                int ixZ = Math.Min(BatchSize, DataSet.Length); //Colocar el último índex en la posición más baja, para evitar error de array lenght
                int numBatch = 0;
                //MIENTRAS EL INDEX INICIAL 'A' SEA MENOR AL LÍMITE DEL ARRAY DE TODAS LAS IMÁGENES
                while (ixA < DataSet.Length || !DeadLineReached)
                {
                    int ActualGeneration = 0;
                    int GenNewVals = EachGensNewVals;
                    // RandomHelper.GetRandomInt(0, FunctionStatus.Length);
                    int ixFunctionK = 0;
                    int ixFunctionW = 0;
                    ////PROCESAR LA POBLACIÓN INICIAL DE VALORES
                    for (int ixI = 0; ixI < InitialPop; ixI++)
                    {
                        //CALCULAR LA APTITUD DE LA POBLACIÓN INICIAL
                        double ErrorIndividual = 0;
                        for (int ixDS = ixA; ixDS < ixZ; ixDS++)
                        {
                            //Implementar cada una de las capas propuestas en la sección convolutiva
                            double[] values = CNNModel.FillAndProcess(EvoIArray[ixI].KValues, DataSet[ixDS].Values_Jagged);
                            //Implementar toda la red neuronal Fully Connected y OBTENER EL ERROR
                            double error = NNModel.Process(EvoIArray[ixI].WValues, values, DataSet[ixDS].Expected);
                            ErrorIndividual += error;
                        }
                        EvoIArray[ixI].Error = ErrorIndividual / (ixZ - ixA);
                    }
                    double ActualError = 1;
                    //MIENTRAS NO SE HAYA ENCONTRADO UNA SOLUCIÓN SATISFACTORIA PARA EL BATCH ACTUAL
                    double BestErrorOfAll = 10;
                    while ((ActualError > ErrorLimit && ActualGeneration < MaxGens) && !DeadLineReached)
                    {
                        //var timerGeneration = System.Diagnostics.Stopwatch.StartNew();
                        string funK = "";
                        string funW = "";
                        //SI LA GENERACIÓN ACTUAL ALCANZÓ UN NÚMERO EN PARTICULAR
                        if (ActualGeneration == GenNewVals)
                        {
                            //REEMPLZAR PARTE DE LA PRIMERA MITAD CON VALORES TOTALMENTE NUEVOS
                            int lastPosInitialPop = InitialPop - SizeNewVals;
                            for (int ixN = 0; ixN < SizeNewVals; ixN++)
                            {
                                //CREAR UNA POBLACIÓN TOTALMENTE NUEVA
                                EvoIArray[lastPosInitialPop + ixN] = new EvoWIndividual();
                                EvoIArray[lastPosInitialPop + ixN].KValues = new double[totalValKernel];
                                for (int ix = 0; ix < totalValKernel; ix++)
                                    EvoIArray[lastPosInitialPop + ixN].KValues[ix] = RandomHelper.GetRandomDouble(MinKVal, MaxKVal, DecimalsKVal);
                                //Pesos para conexiones neuronales (NN)
                                EvoIArray[lastPosInitialPop + ixN].WValues = new double[totalValWeight];
                                for (int ix = 0; ix < totalValWeight; ix++)
                                    EvoIArray[lastPosInitialPop + ixN].WValues[ix] = RandomHelper.GetRandomDouble(MinWVal, MaxWVal, DecimalsWVal);
                                EvoIArray[lastPosInitialPop + ixN].ID = lastPosInitialPop + ixN;
                            }
                            GenNewVals += EachGensNewVals;
                            //Console.WriteLine($"[{EAI.ID}]SE HAN AGREGADO {SizeNewVals} VALORES COMPLETAMENTE NUEVOS--------------------------");
                            ////PROCESAR LA POBLACIÓN INICIAL DE VALORES
                            for (int ixI = 0; ixI < InitialPop; ixI++)
                            {
                                //CALCULAR LA APTITUD DE LA POBLACIÓN INICIAL
                                double ErrorIndividual = 0;
                                for (int ixDS = ixA; ixDS < ixZ; ixDS++)
                                {
                                    //Implementar cada una de las capas propuestas en la sección convolutiva
                                    double[] values = CNNModel.FillAndProcess(EvoIArray[ixI].KValues, DataSet[ixDS].Values_Jagged);
                                    //Implementar toda la red neuronal Fully Connected y OBTENER EL ERROR
                                    double error = NNModel.Process(EvoIArray[ixI].WValues, values, DataSet[ixDS].Expected);
                                    ErrorIndividual += error;
                                }
                                EvoIArray[ixI].Error = ErrorIndividual / (ixZ - ixA);
                            }
                        }
                        //VOLVER A POBLAR LOS PESOS PARA LAS MUTACIONES POR EVOLUCIÓN DIFERENCIAL RAND/3 y RAND/BEST
                        for (int ixI = 0; ixI < InitialPop; ixI++)
                        {
                            InitialPopK[ixI] = EvoIArray[ixI].KValues;
                            InitialPopW[ixI] = EvoIArray[ixI].WValues;
                        }
                        //SELECTOR METAHEURÍSTICO
                        double probUseFunction;
                        double accFunction;
                        do
                        {
                            ixFunctionK = RandomHelper.GetRandomInt(0, FunctionStatusK.Length);
                            probUseFunction = RandomHelper.GetRandomDouble(0, 1);
                            accFunction = FunctionStatusK[ixFunctionK].GetAccuracy();
                        }
                        while (probUseFunction > accFunction);
                        funK = $"{(FunctionStatusK[ixFunctionK].GetAccuracy() * 100)}%";
                        do
                        {
                            ixFunctionW = RandomHelper.GetRandomInt(0, FunctionStatusW.Length);
                            probUseFunction = RandomHelper.GetRandomDouble(0, 1);
                            accFunction = FunctionStatusW[ixFunctionW].GetAccuracy();
                        }
                        while (probUseFunction > accFunction);
                        funW = $"{(FunctionStatusW[ixFunctionW].GetAccuracy() * 100)}%";
                        FunctionStatusK[ixFunctionK].Calls++;
                        FunctionStatusW[ixFunctionW].Calls++;
                        #region MUTACIÓN Y CÁLCULO DE APTITUD
                        Task<EvoWIndividual>[] ProcessMutants = new Task<EvoWIndividual>[InitialPop];
                        //PARA CADA INDIVIDUO DE LA POBLACIÓN INICIAL
                        for (int ixI = 0; ixI < InitialPop; ixI++)
                        {
                            EvoIArray[ixI].Victories = 0;
                            ProcessMutants[ixI] = MutateAndProcessEWI(ixI + InitialPop, EvoIArray[ixI].KValues, EvoIArray[ixI].WValues, EAI.CLayers, EAI.NLayers, ixFunctionK, ixFunctionW, MinKVal, MaxKVal, DecimalsKVal, MinWVal, MaxWVal, DecimalsWVal, MinFVal, MaxFVal, DecimalsFVal, InitialPopK, InitialPopW, ixI, ixA, ixZ, DataSet, imgSizeIn, imgDimensionIn, DeadLine);
                        }
                        //Esperar al resultado de todos los procesos
                        Task<EvoWIndividual[]> CombinedTasksMutants = Task.WhenAll(ProcessMutants);
                        EvoWIndividual[] ResultsMutants = CombinedTasksMutants.Result;
                        //Por cada uno de los resultados
                        for (int ixTI = 0; ixTI < ResultsMutants.Length; ixTI++)
                        {
                            //Actualizar el Hijo
                            int ixMut = ResultsMutants[ixTI].ID;
                            EvoIArray[ixMut] = ResultsMutants[ixTI];
                        }
                        //var x = 1;
                        #endregion
                        #region TORNEO Y SELECCIÓN
                        //REALIZAR EL TORNEO ESTOCÁSTICO ENTRE TODA LA POBLACIÓN
                        for (int ixWT = 0; ixWT < TotalPop; ixWT++)
                        {
                            r = new Random();
                            //...Realizar n cantidad de competiciones aleatorias entre miembros de la población total
                            for (int t = 0; t < TotalComp; t++)
                            {
                                int IxComp;
                                //Obtener un index de competición (IxComp) aleatorio que sea diferente del index de pesos actual (ixW)
                                do
                                    IxComp = r.Next(0, TotalPop - 1);
                                while (ixWT == IxComp);
                                if (EvoIArray[ixWT].Error < EvoIArray[IxComp].Error)
                                    EvoIArray[ixWT].Victories++;
                            }
                        }
                        //ORDENAR LA POBLACIÓN DE PESOS CONFORME A LAS VICTORIAS
                        EvoIArray = EvoIArray.OrderByDescending(w => w.Victories).ToArray();
                        //LA MITAD MÁS ALTA SERÁ LA NUEVA POBLACIÓN INICIAL
                        for (int ixE = 0; ixE < EvoIArray.Length; ixE++)
                            EvoIArray[ixE].ID = ixE;
                        ActualError = EvoIArray[0].Error;
                        if (ActualError < BestErrorOfAll)
                        {
                            //El nuevo es mejor (al ser menor), por lo tanto hay que darle un punto al método que lo hizo posible
                            //INGRESAR LA VICTORIA A LA FUNCIÓN EN LA ÚLTIMA POSICIÓN SELECCIONADA
                            FunctionStatusK[ixFunctionK].Hits++;
                            FunctionStatusW[ixFunctionW].Hits++;
                            BestErrorOfAll = ActualError;
                        }
                        //PreviousErrror = ActualError;
                        BestIndividual = EvoIArray[0];
                        ActualGeneration++;
                        TotalGensPerEpoch++;
                        //timerGeneration.Stop();
                        //var TiempoEntrenamiento = timerGeneration.Elapsed;
                        //Console.WriteLine($"E: {ixEpoch} |B: {numBatch} |B TAM.: {ixA} - {ixZ} |GEN.: {ActualGeneration} |LÍM. ER.: {LE} |MIN. ER.: {ActualError} | FK:{funK} |FW:{funW}");
                        //Console.WriteLine($"E: {ixEpoch} |B: {numBatch} [{ixA} - {ixZ}] |GEN.: {ActualGeneration} | {ErrorLimit} -> {ActualError} - {EvoIArray[InitialPop - 1].Error} |{fKN}({funK}) - {fWN}({funW})");
                        //Console.WriteLine($"E: {ixEpoch} |B: {numBatch} [{ixA} - {ixZ}] |GEN.: {ActualGeneration} | {ErrorLimit} -> {ActualError} - {EvoIArray[InitialPop - 1].Error} | {TiempoEntrenamiento}");
                        DeadLineReached = DateTime.Now > DeadLine;
                        #endregion
                    }//FIN-MIENTRAS
                    //UNA VEZ SALIDO DEL CICLO, AUMENTAR LOS INDEX A Y Z TANTO COMO EL TAMAÑO DEL BATCH
                    ixA += BatchSize;
                    ixZ += BatchSize;
                    //EN CASO DE QUE EL INDEX Z SEA MAYOR AL TOTAL DE IMÁGENES, SUSTITUIRLA POR ESE NÚMERO
                    ixZ = Math.Min(ixZ, DataSet.Length);
                    //Console.WriteLine($"# EPOCH: {ixEpoch} | # BATCH: {numBatch} | CURRENT BATCH SIZE: {ixZ} - {ixA} | {(string.Join(", ", (FunctionStatusK.Select(fun => $"{fun.Hits}/{fun.Calls}").ToArray())))}");
                    if (ActualGeneration < MaxGens)
                    {
                        //Se alcanzó la meta antes del tope de generaciones
                        NoHaltByGenerationLimit++;
                    }
                    numBatch++;
                    TotalBatches++;
                }

                //FIN-MIENTRAS
                //REORDENAR TODO EL UNIVERSO DE IMÁGENES DE FORMA ALEATORIA PARA EL SIGUIENTE EPOCH
                //Utilizaré Linq por rapidez
                r = new Random();
                DataSet = DataSet.OrderBy(d => r.Next()).ToArray();
                timerEpoch.Stop();
                var timeEpoch = timerEpoch.Elapsed;
                double AvrgGens = 0;
                try
                {
                    AvrgGens = Math.Round(TotalGensPerEpoch / Convert.ToDouble(TotalBatches), 2);
                }
                catch (Exception) { }
                ResumeMetaSelector[ixEpoch] = $"{ixEpoch}° Epoch|{TotalBatches} Batchs|{NoHaltByGenerationLimit} Batchs sin topar límite de gens|{timeEpoch.TotalMinutes} Minutos del Epoch|{AvrgGens} Gens/Batchs|Deadline:{DeadLineReached}";
                Console.WriteLine($"#[ {EAI.ID} ] RESUME: " + ResumeMetaSelector[ixEpoch]);
                ixEpoch++;
            }
            CNNModel.FillWeights(BestIndividual.KValues);
            NNModel.FillWeights(BestIndividual.WValues);
            TrainnedCNNModel = CNNModel;
            TrainnedNNModel = NNModel;
            timerAllProcess.Stop();
            var TotalTime = timerAllProcess.Elapsed;
            TotalMinutesRequired = TotalTime.Minutes;
        }
        /// <summary>
        /// T-IV. Método que se encarga de obtener un mutante a partir de los datos de un individuo de pesos y realizar su cálculo de aptitud (porcentaje de error en el entrenamiento). Todo dentro de un Task (Hilo)
        /// </summary>
        /// <param name="posMutate">Posición dentro del vector de individuos donde se implantará el mutante</param>
        /// <param name="kVals">Vector de pesos de Kernel (convolutivo) del 'padre'</param>
        /// <param name="wVals">Vector de pesos sinápticos (neuronal) del 'padre'</param>
        /// <param name="CNNLayers">Capas que componen a la sección convolutiva del modelo</param>
        /// <param name="NNLayers">'Capas' que componen a la sección neuronal del modelo</param>
        /// <param name="ixFunctionK">Posición elegida por el selector metaheurístico del vector de funciones para mutar el vector de pesos de Kernel del 'padre'</param>
        /// <param name="ixFunctionW">Posición elegida por el selector metaheurístico del vector de funciones para mutar el vector de pesos sinápticos del 'padre'</param>
        /// <param name="MinKVal">Valor mínimo para un nuevo valor de peso de Kernel</param>
        /// <param name="MaxKVal">Valor máximo para un nuevo valor de peso de Kernel</param>
        /// <param name="DecimalsKVal">Cantidad máxima de decimales para un nuevo valor de peso de Kernel</param>
        /// <param name="MinWVal">Valor mínimo para un nuevo valor de peso sináptico</param>
        /// <param name="MaxWVal">Valor máximo para un nuevo valor de peso sináptico</param>
        /// <param name="DecimalsWVal">Cantidad máxima de decimales para un nuevo valor de peso sináptico</param>
        /// <param name="MinFVal">Valor mínimo para la constante F de evolución diferencial</param>
        /// <param name="MaxFVal">Valor máximo para la constante F de evolución diferencial</param>
        /// <param name="DecimalsFVal">Cantidad máxima de decimales para la constante F de evolución diferencial</param>
        /// <param name="InitialPopK">Arreglo del total de vectores de pesos convolutivos (para evolución diferencial)</param>
        /// <param name="InitialPopW">Arreglo del total de vectores de pesos sinápticos (para evolución diferencial)</param>
        /// <param name="ixI">Índice del vector 'padre' para su consideración dentro de evolución diferencial</param>
        /// <param name="ixA">Índice de inicio del Batch (imagen dentro del dataset)</param>
        /// <param name="ixZ">Índice final del Batch (imagen dentro del dataset)</param>
        /// <param name="DataSet">Conjunto de imágenes para el entrenamiento</param>
        /// <param name="imgSizeIn">Tamaño estándar de las imágenes de entrada (ancho o alto)</param>
        /// <param name="imgDimensionIn">Dimensión estándar de las imágenes de entrada (canales RGB o Grises)</param>
        /// <param name="DeadLine">Tiempo límite (en fecha) para el entrenamiento</param>
        /// <returns></returns>
        private static Task<EvoWIndividual> MutateAndProcessEWI(int posMutate, double[] kVals, double[] wVals,
            ConvLayer[] CNNLayers, int[] NNLayers,
            int ixFunctionK, int ixFunctionW,
            double MinKVal, double MaxKVal, int DecimalsKVal,
            double MinWVal, double MaxWVal, int DecimalsWVal,
            double MinFVal, double MaxFVal, int DecimalsFVal,
            double[][] InitialPopK, double[][] InitialPopW, int ixI,
            int ixA, int ixZ, Data[] DataSet, int imgSizeIn, int imgDimensionIn, DateTime DeadLine
            )
        {
            return Task.Factory.StartNew(() =>
            {
                //Al ser la ejecución de un hilo, este debe tener su propia instancia de Modelos Convolutivos y Neuronales
                //para evitar que varios hilos accedan al mismo tiempo a los modelos y alteren el resultado de forma no controlada
                //ConvLayer[] CNNLayers = EAI.CLayers;
                //Construir el modelo nuevamente
                ConvolutionNetwork CNNModelPrime = new ConvolutionNetwork();
                int x, y;
                ConvLayer[] CNNLayerPrime = new ConvLayer[CNNLayers.Length];
                for (int ixC = 0; ixC < CNNLayers.Length; ixC++)
                {
                    if (!CNNLayers[ixC].HasPooling())
                        CNNLayerPrime[ixC] = new ConvLayer(CNNLayers[ixC].GetTotalKernels(), CNNLayers[ixC].GetKernelSize(), CNNLayers[ixC].GetKernelStride(), CNNLayers[ixC].GetPadding());
                    else
                        CNNLayerPrime[ixC] = new ConvLayer(CNNLayers[ixC].GetTotalKernels(), CNNLayers[ixC].GetKernelSize(), CNNLayers[ixC].GetKernelStride(), CNNLayers[ixC].GetPadding(), CNNLayers[ixC].GetPoolingSize(), CNNLayers[ixC].GetPoolingStride());
                }
                //Array.Copy(CNNLayers, CNNLayerPrime, CNNLayers.Length);
                int totalKernelValues = CNNModelPrime.Build(CNNLayerPrime, imgSizeIn, imgDimensionIn, out x, out y);
                NeuralNetwork NNModelPrime = new NeuralNetwork();
                NeuralNetwork.ErrorMethod errorMethod = NeuralNetwork.ErrorMethod.MeanSquaredError;
                int totalWeigthsValues = NNModelPrime.Build(NNLayers, PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal,
                    PropagationRule.Lineal, ActivationFunction.Hiperbolic, OutputFunction.Lineal, errorMethod);
                //Console.WriteLine($"ELEMENTO MUTADO #[{posMutate}] - MODELOS CREADOS");
                //Realizar lo siguiente
                EvoWIndividual result = new EvoWIndividual();
                result.ID = posMutate;
                //REALIZACIÓN DE LA MUTACIÓN
                double F;
                switch (ixFunctionK)
                {
                    case 0: //Sustituye TODOS los valores de pesos con nuevos
                        result.KValues = AbsoluteMutateWeights(kVals, MinKVal, MaxKVal, DecimalsKVal);
                        break;
                    case 1: //Evolución Diferencial RAND/3 con F Aleatorio
                        F = RandomHelper.GetRandomDouble(MinFVal, MaxFVal, DecimalsFVal);
                        result.KValues = DifferentialEvolution_Rand3(InitialPopK, MinKVal, MaxKVal, ixI, F);
                        break;
                    case 2: //Evolución Diferencial BEST/3 con F Aleatorio
                        F = RandomHelper.GetRandomDouble(MinFVal, MaxFVal, DecimalsFVal);
                        result.KValues = DifferentialEvolution_RandBest3(InitialPopK, MinKVal, MaxKVal, ixI, F);
                        break;
                    case 3: //Mediante un FACTOR aleatorio (-/+) modifica los pesos cada X posiciones
                        result.KValues = StaticMutateWeights(kVals, MinKVal, MaxKVal, DecimalsKVal);
                        break;
                    case 4: //Sustituye 1/3 de los pesos en posiciones totalmente aleatorias
                        result.KValues = RandomMutateWeights(kVals, MinKVal, MaxKVal, DecimalsKVal);
                        break;
                    case 5: //Sustituye 1/5 de los pesos en posiciones totalmente aleatorias
                        result.KValues = RandomMutateWeights2(kVals, MinKVal, MaxKVal, DecimalsKVal);
                        break;
                    case 6: //Evolución Diferencial RAND/3 con F estático de 0.8
                        result.KValues = DifferentialEvolution_Rand3_Fs(InitialPopK, MinKVal, MaxKVal, ixI);
                        break;
                }
                switch (ixFunctionW)
                {
                    case 0: //Sustituye TODOS los valores de pesos con nuevos
                        result.WValues = AbsoluteMutateWeights(wVals, MinWVal, MaxWVal, DecimalsWVal);
                        break;
                    case 1: //Evolución Diferencial RAND/3 con F Aleatorio
                        F = RandomHelper.GetRandomDouble(MinFVal, MaxFVal, DecimalsFVal);
                        result.WValues = DifferentialEvolution_Rand3(InitialPopW, MinWVal, MaxWVal, ixI, F);
                        break;
                    case 2: //Evolución Diferencial BEST/3 con F Aleatorio
                        F = RandomHelper.GetRandomDouble(MinFVal, MaxFVal, DecimalsFVal);
                        result.WValues = DifferentialEvolution_RandBest3(InitialPopW, MinWVal, MaxWVal, ixI, F);
                        break;
                    case 3: //Mediante un FACTOR aleatorio (-/+) modifica los pesos cada X posiciones
                        result.WValues = StaticMutateWeights(wVals, MinWVal, MaxWVal, DecimalsWVal);
                        break;
                    case 4: //Sustituye 1/3 de los pesos en posiciones totalmente aleatorias
                        result.WValues = RandomMutateWeights(wVals, MinWVal, MaxWVal, DecimalsWVal);
                        break;
                    case 5: //Sustituye 1/5 de los pesos en posiciones totalmente aleatorias
                        result.WValues = RandomMutateWeights2(wVals, MinWVal, MaxWVal, DecimalsWVal);
                        break;
                    case 6: //Evolución Diferencial RAND/3 con F estático de 0.8
                        result.WValues = DifferentialEvolution_Rand3_Fs(InitialPopW, MinWVal, MaxWVal, ixI);
                        break;
                }
                double ErrorMutated = 0;
                bool DeadLineReached = false;
                //Pre-DeadLine;
                //for (int ixDS = ixA; ixDS < ixZ; ixDS++)
                //Post-DeadLine;
                int ixDS = ixA;
                while(ixDS < ixZ && !DeadLineReached)
                {
                    //Implementar cada una de las capas propuestas en la sección convolutiva
                    double[] values = CNNModelPrime.FillAndProcess(result.KValues, DataSet[ixDS].Values_Jagged);
                    //Implementar toda la red neuronal Fully Connected y OBTENER EL ERROR
                    ErrorMutated += NNModelPrime.Process(result.WValues, values, DataSet[ixDS].Expected);
                    ixDS++;
                    DeadLineReached = DateTime.Now > DeadLine;
                }
                if(DeadLineReached && ixDS < ixZ)
                {
                    //Se alcanzó el limite de tiempo y no se completó el proceso para cada imagen
                    //por lo que se sumarán como enteros al error
                    ErrorMutated += ixZ - ixDS;
                }
                result.Error = ErrorMutated / (ixZ - ixA);
                return result;
            });
        }
        /// <summary>
        /// T-V. Método auxiliar para probar los modelos en la clasificación de imágenes. Retorna el número de aciertos (hits)
        /// </summary>
        /// <param name="TestDataSet">Conjunto de imágenes de prueba</param>
        /// <param name="CNNModel">Sección convolutiva del modelo (entrenado)</param>
        /// <param name="NNModel">Sección neuronal del modelo (entrenado)</param>
        /// <returns></returns>
        public static int Hyperheuristic_Testing(Data[] TestDataSet, ConvolutionNetwork CNNModel, NeuralNetwork NNModel)
        {
            Prediction prediction;
            int hits = 0;
            for (int ixD = 0; ixD < TestDataSet.Length; ixD++)
            {
                Data Data4Test = TestDataSet[ixD];
                Data4Test.Values = CNNModel.ProcessImage(Data4Test.Values_Jagged);
                prediction = NNModel.Predict(Data4Test);
                if (prediction.hit)
                    hits++;
            }
            return hits;
        }

        #region MÉTODOS HEURÍSTICOS SIMPLES
        /// <summary>
        /// Heurística - Evolución Diferencial con Rand/3 con F variable
        /// </summary>
        /// <param name="pop">Arreglo con la totalidad de vectores de pesos (convolutivos o neuronales)</param>
        /// <param name="min">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="max">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="ixActualIndividual">Index que tiene el vector 'padre' dentro del arreglo 'pop'</param>
        /// <param name="F">Valor que tomará el factor F</param>
        /// <returns></returns>
        private static double[] DifferentialEvolution_Rand3(double[][] pop, double min, double max, int ixActualIndividual, double F)
        {
            const double C = 0.5; //Factor de elección en la decisión aleatoria FlipCoin.
            Random r = new Random();
            int popK = pop[0].Length;
            double[] newVals = new double[popK];

            int[] RPos = { -1, -1, -1 }; //Para que sea posible obtener un índice 0
            int rTemp; //Para almacenar temporalmente el valor random
            double Coin;
            //Mutate Kernel---------------------------------
            //I. Generar 3 números aleatorios para las posiciones.
            //Estos números deben ser mutuamente excluyentes y no debe ser el índice (i) actual
            for (int ixR = 0; ixR < 3; ixR++)
            {
                do
                    rTemp = r.Next(pop.Length);
                while (RPos.Contains(rTemp) || rTemp == ixActualIndividual);
                RPos[ixR] = rTemp;
            }
            //II. Evolución diferencial 
            //Obtenidas las 3 posiciones, crear cada uno de los valores para generar el vector evolucionado de pesos
            for (int ixK = 0; ixK < popK; ixK++)
            {
                //III. Flip coin
                //Una vez creado el valor mutados, realizar el 'volteo de moneda' (flip coin), 
                //para decidir la elección de los valores del vector final de pesos
                Coin = RandomHelper.GetRandomDouble(0, 1);
                if (Coin <= C) //Colocar en la posición el valor creado
                {
                    //Vi = Wr0[i]+F(Wr1[i] - Wr2[i]) (DE/Rand/3)
                    double result = Math.Round((pop[RPos[0]][ixK] + (F * (pop[RPos[1]][ixK] - pop[RPos[2]][ixK]))), 3);
                    newVals[ixK] = result;
                }
                else           //Colocar en la posición el valor original
                    newVals[ixK] = pop[ixActualIndividual][ixK];
            }
            newVals = RandomHelper.Reflexion(newVals, min, max);
            return newVals;
        }
        /// <summary>
        /// Heurística - Evolución Diferencial con Best/3 con F variable
        /// </summary>
        /// <param name="pop">Arreglo con la totalidad de vectores de pesos (convolutivos o neuronales)</param>
        /// <param name="min">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="max">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="ixActualIndividual">Index que tiene el vector 'padre' dentro del arreglo 'pop'</param>
        /// <param name="F">Valor que tomará el factor F</param>
        /// <returns></returns>
        private static double[] DifferentialEvolution_RandBest3(double[][] pop, double min, double max, int ixActualIndividual, double F)
        {
            const double C = 0.5; //Factor de elección en la decisión aleatoria FlipCoin.
            Random r = new Random();
            int popK = pop[0].Length;

            double[] newVals = new double[popK];

            int[] RPos = { -1, -1 }; //Para que sea posible obtener un índice 0
            int rTemp; //Para almacenar temporalmente el valor random
            double Coin;
            //Mutate Kernel---------------------------------
            //I. Generar 3 números aleatorios para las posiciones.
            //Estos números deben ser mutuamente excluyentes y no debe ser el índice (i) actual
            for (int ixR = 0; ixR < 2; ixR++)
            {
                do
                    rTemp = r.Next(pop.Length);
                while (RPos.Contains(rTemp) || rTemp == ixActualIndividual);
                RPos[ixR] = rTemp;
            }
            //II. Evolución diferencial 
            //Obtenidas las 3 posiciones, crear cada uno de los valores para generar el vector evolucionado de pesos
            for (int ixK = 0; ixK < popK; ixK++)
            {
                //III. Flip coin
                //Una vez creado el valor mutados, realizar el 'volteo de moneda' (flip coin), 
                //para decidir la elección de los valores del vector final de pesos
                Coin = RandomHelper.GetRandomDouble(0, 1);
                if (Coin <= C) //Colocar en la posición el valor creado
                {
                    //Vi = Wbest[i]+F(Wr0[i] - Wr1[i]) (DE/Best/3)
                    double result = Math.Round((pop[0][ixK] + (F * (pop[RPos[0]][ixK] - pop[RPos[1]][ixK]))), 3);
                    newVals[ixK] = result;
                }
                else           //Colocar en la posición el valor original
                    newVals[ixK] = pop[ixActualIndividual][ixK];
            }
            newVals = RandomHelper.Reflexion(newVals, min, max);

            return newVals;
        }
        /// <summary>
        /// Heurística - Mutar a parte de la población con números aleatorios, en posiciones aleatorias
        /// </summary>
        /// <param name="origin">Vector 'padre' de pesos (convolutivos o neuronales)</param>
        /// <param name="MinVal">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="MaxVal">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="numDec">Cantidad de decimales que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <returns></returns>
        private static double[] AbsoluteMutateWeights(double[] origin, double MinVal, double MaxVal, int numDec)
        {

            Random r = new Random();
            double[] news = new double[origin.Length];
            for (int ixW = 0; ixW < news.Length; ixW++)
            {
                news[ixW] = RandomHelper.GetRandomDouble(MinVal, MaxVal, numDec);
            }
            return news;
        }
        /// <summary>
        /// Heurística - Mutar a parte de la población con números aleatorios, en posiciones aleatorias
        /// </summary>
        /// <param name="origin">Vector 'padre' de pesos (convolutivos o neuronales)</param>
        /// <param name="MinVal">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="MaxVal">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="numDec">Cantidad de decimales que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <returns></returns>
        private static double[] StaticMutateWeights(double[] origin, double MinVal, double MaxVal, int numDec)
        {
            Random r = new Random();
            double Factor = RandomHelper.GetRandomDouble(MinVal, MaxVal, numDec);
            //int Size = origin.Length / 200;
            //if (Size == 0)
            //Size = r.Next(0, origin.Length - 1);
            int Size = origin.Length <= 5 ? origin.Length : 5;
            int Pivot = r.Next(0, Size); //r.Next(0, origin.Length - 1);
            double[] news = new double[origin.Length];
            Array.Copy(origin, news, origin.Length);
            while (Pivot < origin.Length)
            {
                news[Pivot] += Factor;
                if (Pivot == 0)
                {
                    //Pivot = r.Next(1, origin.Length - 1);
                    Pivot = r.Next(1, Size);
                }
                Pivot += Pivot;
            }
            return news;
        }
        /// <summary>
        /// Heurística - Mutar a gran parte de la población (max 1/3) con números aleatorios, en posiciones aleatorias
        /// </summary>
        /// <param name="origin">Vector 'padre' de pesos (convolutivos o neuronales)</param>
        /// <param name="minV">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="maxV">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="dec">Cantidad de decimales que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <returns></returns>
        private static double[] RandomMutateWeights(double[] origin, double minV, double maxV, int dec)
        {
            double times = 0;
            times = Math.Ceiling(Convert.ToDouble(origin.Length) / 3.0);
            double[] news = new double[origin.Length];
            Array.Copy(origin, news, origin.Length);
            Random r = new Random();
            while (times > 0)
            {
                news[r.Next(0, origin.Length - 1)] = RandomHelper.GetRandomDouble(r, minV, maxV, dec);
                times--;
            }
            return news;
        }
        /// <summary>
        /// Heurística - Mutar a parte de la población (max 1/5) con números aleatorios, en posiciones aleatorias
        /// </summary>
        /// <param name="origin">Vector 'padre' de pesos (convolutivos o neuronales)</param>
        /// <param name="minV">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="maxV">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="dec">Cantidad de decimales que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <returns></returns>
        private static double[] RandomMutateWeights2(double[] origin, double minV, double maxV, int dec)
        {
            double times = 0;
            times = Math.Ceiling(Convert.ToDouble(origin.Length) / 5.0);
            double[] news = new double[origin.Length];
            Array.Copy(origin, news, origin.Length);
            Random r = new Random();
            while (times > 0)
            {
                news[r.Next(0, origin.Length - 1)] = RandomHelper.GetRandomDouble(r, minV, maxV, dec);
                times--;
            }
            return news;
        }
        /// <summary>
        /// Heurística - Evolución Diferencial con Rand/3 con F fijo en 0.8
        /// </summary>
        /// <param name="pop">Arreglo con la totalidad de vectores de pesos (convolutivos o neuronales)</param>
        /// <param name="min">Valor mínimo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="max">Valor máximo que tendrá cada peso del vector mutante o 'hijo'</param>
        /// <param name="ixActualIndividual">Index que tiene el vector 'padre' dentro del arreglo 'pop'</param>
        /// <returns></returns>
        private static double[] DifferentialEvolution_Rand3_Fs(double[][] pop, double min, double max, int ixActualIndividual)
        {
            const double F = 0.8; //Factor de multiplicación que interactuará con el peso resultante.
            const double C = 0.5; //Factor de elección en la decisión aleatoria FlipCoin.
            Random r = new Random();
            int popK = pop[0].Length;

            double[] newVals = new double[popK];

            int[] RPos = { -1, -1, -1 }; //Para que sea posible obtener un índice 0
            int rTemp; //Para almacenar temporalmente el valor random
            double Coin;
            //Mutate Kernel---------------------------------
            //I. Generar 3 números aleatorios para las posiciones.
            //Estos números deben ser mutuamente excluyentes y no debe ser el índice (i) actual
            for (int ixR = 0; ixR < 3; ixR++)
            {
                do
                    rTemp = r.Next(pop.Length);
                while (RPos.Contains(rTemp) || rTemp == ixActualIndividual);
                RPos[ixR] = rTemp;
            }
            //II. Evolución diferencial 
            //Obtenidas las 3 posiciones, crear cada uno de los valores para generar el vector evolucionado de pesos
            for (int ixK = 0; ixK < popK; ixK++)
            {
                //III. Flip coin
                //Una vez creado el valor mutados, realizar el 'volteo de moneda' (flip coin), 
                //para decidir la elección de los valores del vector final de pesos
                Coin = RandomHelper.GetRandomDouble(0, 1);
                if (Coin <= C) //Colocar en la posición el valor creado
                {
                    //Vi = Wr0[i]+F(Wr1[i] - Wr2[i]) (DE/Rand/3)
                    double result = Math.Round((pop[RPos[0]][ixK] + (F * (pop[RPos[1]][ixK] - pop[RPos[2]][ixK]))), 3);
                    newVals[ixK] = result;
                }
                else           //Colocar en la posición el valor original
                    newVals[ixK] = pop[ixActualIndividual][ixK];
            }
            newVals = RandomHelper.Reflexion(newVals, min, max);

            return newVals;
        }
        #endregion
    }
}
