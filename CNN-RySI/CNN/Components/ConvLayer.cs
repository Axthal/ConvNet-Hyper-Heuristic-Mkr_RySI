using System;
namespace CNN_RySI.CNN.Components
{
    /// <summary>
    /// Capa de convolución. Se tomará toda la capa como si fuera convolutiva.
    /// Adicionalmente se tomará solamente el ReLu como función de "activación" al ser el mejor hasta el momento
    /// En caso de que se especifiquen valores de Pooling, se realizará el cálculo.
    /// El pooling será de valores Máximos, ya que se ha probado en la literatura como el mejor hasta el momento
    /// </summary>
    public class ConvLayer
    {
        //Datos del Kernel 
        private int kernelSize;
        private int totalKernels;
        /// <summary>
        /// [Número de Kernel][Dimensión][Fila][Columna]
        /// </summary>
        private double[][][][] kernels;
        private int dimensionsPerKernel;
        private int strideK;
        private int paddingK;
        //Datos del pooling
        private int poolingSize;
        private int strideP;
        private bool hasPooling;
        //Almacenará el ancho-alto de la imagen que salga del proceso convolutivo
        private int outSizeKernel;
        //Almacenará el ancho-alto de la imagen que salga de la capa (que puede o no ser el mismo valor que el putSizeKernel,
        //pues la capa puede tener un pooling)
        private int outSizeLayer;
        public ConvLayer(int totalKernels, int kSize, int kStride, int kPadding)
        {
            kernelSize = kSize;
            strideK = kStride;
            paddingK = kPadding;
            this.totalKernels = totalKernels;
            hasPooling = false;
            kernels = new double[totalKernels][][][];
        }
        public ConvLayer(int totalKernels, int kSize, int kStride, int kPadding, int pSize, int pStride)
        {
            kernelSize = kSize;
            strideK = kStride;
            paddingK = kPadding;
            this.totalKernels = totalKernels;
            if (pSize > 0)
            {
                hasPooling = true;
                poolingSize = pSize;
                strideP = pStride;
            }
            else
                hasPooling = false;
            kernels = new double[totalKernels][][][];
        }
        public int GetKernelSize()
        {
            return kernelSize;
        }
        public int GetTotalKernels()
        {
            return totalKernels;
        }
        public int GetKernelStride()
        {
            return strideK;
        }
        public int GetPadding()
        {
            return paddingK;
        }
        public int GetPoolingSize()
        {
            return poolingSize;
        }
        public int GetPoolingStride()
        {
            return strideP;
        }
        public bool HasPooling()
        {
            return hasPooling;
        }
        /// <summary>
        /// Determina si la capa puede procesar de forma satisfactoria con los parámetros datos
        /// Para la validación no es importante el número de canales o dimensiones de entrada
        /// Sin embargo, si la configuración es válida, se creará la estructura de Kernels (ya que al ser objetos, por defecto son null)
        /// </summary>
        /// <param name="inputSize"></param>
        /// <param name="inputChannels"></param>
        /// <param name="outSize"></param>
        /// <param name="outChannels"></param>
        /// <returns></returns>
        public bool CanProcessImage(int inputSize, int inputChannels, out int outSize, out int outChannels)
        {
            //Validar si es posible realizar la convolución
            int W = inputSize;
            int K = kernelSize;
            int P = paddingK;
            int S = strideK;
            int Wout = ((W - K + (2 * P)) / S) + 1;
            outSize = Wout;
            outSizeKernel = Wout;
            int tempSum = K;
            for (int ixS = 2; ixS <= Wout; ixS++)
                tempSum += S;
            bool valid = tempSum == (W + P);
            //Si la capa cuenta con proceso de pooling, validar la salida
            if (hasPooling)
            {
                W = outSize;
                K = poolingSize;
                S = strideP;
                int poolW = ((W - K) / S) + 1;
                outSize = poolW;

                tempSum = K;
                for (int ixS = 2; ixS <= poolW; ixS++)
                    tempSum += S;
                valid = valid && (tempSum == W);
            }
            if (valid)
            {
                //Aprovechar y crear la estructura de kernels (filtros) ya que al ser objetos, por defecto son null
                //Se tomará la profundidad de todos los kernels de acuerdo a la profundidad de la entrada
                dimensionsPerKernel = inputChannels;
                //Crear cada uno de los Kernels
                for (int ixN = 0; ixN < totalKernels; ixN++)
                {
                    //Crear un kernel con x canales o dimensiones
                    kernels[ixN] = new double[dimensionsPerKernel][][];
                    //Para cada uno de los canales o dimensiones del Kernel
                    for (int ixD = 0; ixD < dimensionsPerKernel; ixD++)
                    {
                        //Crear el propio canal del kernel
                        kernels[ixN][ixD] = new double[kernelSize][];
                        //Para cada una de las filas del canal del kernel
                        for (int ixR = 0; ixR < kernelSize; ixR++)
                        {
                            //Crear el vector de columnas
                            kernels[ixN][ixD][ixR] = new double[kernelSize];
                        }
                    }
                }
            }
            outSizeLayer = outSize;
            //Los canales (o dimensiones) de la imagen de salida serán la totalidad de kernels
            outChannels = totalKernels;
            return valid;
        }
        /// <summary>
        /// Proceso que coloca los valores del vector. Empieza desde la fila 0, columna 0, kernel 0. y va de izquierda a derecha y de arriba a abajo.
        /// </summary>
        /// <param name="values"></param>
        public void FillAllKernels(double[] values)
        {
            if (values.Length != (kernelSize * kernelSize * dimensionsPerKernel * totalKernels))
                throw new Exception("Error. Values lenght isn't same as Kernel elements (n * n * d * t)");
            int ixV = 0;
            //Por cada uno de los Kernels
            for (int ixN = 0; ixN < totalKernels; ixN++)
            {
                //Por cada dimensión o canal del kernel
                for (int ixD = 0; ixD < dimensionsPerKernel; ixD++)
                {
                    //Por cada fila del mismo
                    for (int ixR = 0; ixR < kernelSize; ixR++)
                    {
                        //Por cada columna
                        for (int ixC = 0; ixC < kernelSize; ixC++)
                        {
                            //Colocar el valor
                            kernels[ixN][ixD][ixR][ixC] = values[ixV];
                            ixV++;
                        }
                    }
                }
            }
        }
        /// <summary>
        /// Obtener los valores de los Kernels - [Número de Kernel][Dimensión][Fila][Columna]
        /// </summary>
        /// <returns></returns>
        public double[][][][] GetKernelValues()
        {
            return kernels;
        }
        /// <summary>
        /// Método que obtiene la arquitectura de la capa para fines de informe al usuario
        /// </summary>
        /// <returns></returns>
        public string GetLayerArchitecture()
        {
            string conv = $"{totalKernels} Kernels: ({kernelSize}x{kernelSize}x{dimensionsPerKernel}, Stride {strideK}, Padding {paddingK})";
            string pool = $"Pooling: ({poolingSize}x{poolingSize}, Stride {strideP})";
            return conv + (hasPooling ? " - " + pool : "");
        }
        /// <summary>
        /// Método que obtiene el total de valores de kernel de la capa que deberán ser entrenados
        /// </summary>
        /// <returns></returns>
        public int GetTotalValuesLayer()
        {
            return kernelSize * kernelSize * dimensionsPerKernel * totalKernels;
        }
        /// <summary>
        /// Aplica toda la estructura de la capa en la imagen de entrada. La imagen es un jagged array de 3D [Canal][Fila][Columna]
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[][][] ApplyLayer(double[][][] input)
        {
            //Jagged Array (int[][][][] image = new int[100][][][];) es más rápido, pero ocupa más espacio ya que es un array de array (objetos)
            //Multidimensional Array (int[,] image = new int [100,100];) es más liviano, pero es más lento ya que hace uso de métodos antes de acceder a los arrays.
            //Decidí usar los Jagged Array, ya que no tengo problemas de espacio y requiero poder realizar la convolución rápido.
            //Se da por sentado que ya se validaron los parámetros de la capa

            //La salida tendrá una dimensión del número total de kernels de la capa
            double[][][] output;
            if (paddingK > 0)
            {
                //Obtener el tamaño de la imagen input, recordar que la dimensión es [Canal o Dimension][Filas][Columnas]
                int sizePadd = input[0].Length + (paddingK * 2);
                //Crear el repositorio de la imagen con padding (x2, ya que tendrá ceros antes y después)
                double[][][] imgPadd = new double[input.Length][][];
                //Por cada uno de los canales o dimensiones
                for (int ixD = 0; ixD < imgPadd.Length; ixD++)
                {
                    imgPadd[ixD] = new double[sizePadd][];
                    //Por cada fila del nuevo repositorio
                    for (int ixR = 0; ixR < sizePadd; ixR++)
                    {
                        //Llenar cada fila con ceros (default)
                        imgPadd[ixD][ixR] = new double[sizePadd];
                        //Si el index se encuentra entre el padding y el padding + el tamaño original de la imagen
                        if (ixR >= paddingK && ixR < (paddingK + input[0].Length))
                            //Copiar toda la fila de la imagen original en la fila correspondiente del repositorio
                            //Nota C#: Array.Copy(ArrayOrigen, IndexOrigen, ArrayDestino, IndexDestino, LongitudQueSeCopiará);
                            Array.Copy(input[ixD][ixR - paddingK], 0, imgPadd[ixD][ixR], paddingK, input[ixD][0].Length);
                    }
                }
                output = ProcessConvolution(imgPadd);
            }
            else
            {
                output = ProcessConvolution(input);
            }
            if (hasPooling)
                output = ProcessPooling(output);
            return output;
        }
        /// <summary>
        /// Aplica el proceso de convolución a la imagen o mapa de características proporcionada
        /// </summary>
        /// <param name="input">Valores de la imagen o mapa</param>
        /// <returns></returns>
        private double[][][] ProcessConvolution(double[][][] input)
        {
            double[][][] convOutput = new double[totalKernels][][];
            //Se debe multiplicar la imagen por cada uno de los kernels
            //Por lo que cada una de las dimensiones de la entrada debe multiplicarse por su correspondiente dimensión en cada kernel
            //El resultado de la multiplicación debe sumarse y colocarse en la estructura de salida (convOutput)

            //Por cada uno de los Kernels (funcionará como cada una de las dimensiones del output)
            for (int ixN = 0; ixN < totalKernels; ixN++)
            {
                //Coordenada del input para empezar las mutiplicaciones
                int row = 0, col = 0;
                //Crear en la dimensión ixN de la salida el mapa de activación con filas de tamaño outSizeKernel
                convOutput[ixN] = new double[outSizeKernel][];
                //En cada una de las filas del output
                for (int rowOut = 0; rowOut < outSizeKernel; rowOut++)
                {
                    //Asignarle tantas outSizeKernel columnas a la fila correspondiente
                    convOutput[ixN][rowOut] = new double[outSizeKernel];
                    //En cada una de las columnas del output
                    for (int colOut = 0; colOut < outSizeKernel; colOut++)
                    {
                        //Por cada una de las dimensiones del input (que será la misma dimensión del kernel ixN)
                        for (int ixD = 0; ixD < input.Length; ixD++)
                        {
                            //Procesar la dimensión del input por la dimensión correspondiente del kernel ixN.
                            //Realizar la multiplicación DIRECTA (NO ESCALAR) del kernel por los valores de la imagen que estén en su área
                            //Y almacenarla en la posición correspondiente del output
                            //En cada una de las filas de la dimensión del Kernel
                            for (int rowK = 0; rowK < kernelSize; rowK++)
                                //En cada una de las columnas de la dimensión del Kernel
                                for (int colK = 0; colK < kernelSize; colK++)
                                    //Encontrar el valor del Kernel y multiplicarlo por valor correspondiente en la posición del input
                                    convOutput[ixN][rowOut][colOut] += (kernels[ixN][ixD][rowK][colK] * input[ixD][rowK + row][colK + col]);
                            //Aplicarle de una vez el ReLU al resultado
                            convOutput[ixN][rowOut][colOut] = Math.Max(0, convOutput[ixN][rowOut][colOut]);
                        }
                        //Aumentarle a la coordenada de "col" el stride para que se mueva por la imagen input
                        col += strideK;
                    }
                    //Aumentarle a la coordenada de "row" el stride
                    row += strideK;
                    //Y reiniciar el col a 0
                    col = 0;
                }

            }
            return convOutput;
        }
        /// <summary>
        /// Aplica el proceso de pooling a la imagen o mapa de características proporcionada
        /// </summary>
        /// <param name="input">Valores de la imagen o mapa<</param>
        /// <returns></returns>
        private double[][][] ProcessPooling(double[][][] input)
        {
            double[][][] poolOutput = new double[totalKernels][][];
            //Semejante al proceso de la convolución, solo que se debe recorrer la 'ventana' del kernel en la imagen
            //Y colocar el valor correspondiente en el repositorio, de acuerdo al criterio dado (MAX, que en este caso será fijo)

            //Cada uno de los kernels de convolución será "encogido", por lo que la relación es 1 ConvKernel a 1 PoolKernel
            //Por cada uno de los Kernels de Pooling (funcionará como cada una de las dimensiones del output)
            for (int ixN = 0; ixN < totalKernels; ixN++)
            {
                //Coordenada del input para empezar las mutiplicaciones
                int row = 0, col = 0;
                //Crear en la dimensión ixN de la salida el mapa de activación con filas de tamaño outSizeLayer
                poolOutput[ixN] = new double[outSizeLayer][];
                //En cada una de las filas del output
                for (int rowOut = 0; rowOut < outSizeLayer; rowOut++)
                {
                    //Asignarle tantas outSizeLayer columnas a la fila correspondiente
                    poolOutput[ixN][rowOut] = new double[outSizeLayer];
                    //En cada una de las columnas del output
                    for (int colOut = 0; colOut < outSizeLayer; colOut++)
                    {
                        //Encontrar el MAX valor de toda la ventana
                        double MAX = -1;
                        //En cada una de las filas de la dimensión del Kernel
                        for (int rowP = 0; rowP < poolingSize; rowP++)
                            //En cada una de las columnas de la dimensión del Kernel
                            for (int colP = 0; colP < poolingSize; colP++)
                                //Encontrar el valor máximo dentro de la ventana sobrepuesta del input
                                MAX = Math.Max(MAX, input[ixN][rowP + row][colP + col]);
                        //Asignarle el valor máximo encontrado
                        poolOutput[ixN][rowOut][colOut] = MAX;
                        //Aumentarle a la coordenada de "col" el stride para que se mueva por la imagen input
                        col += strideP;
                    }
                    //Aumentarle a la coordenada de "row" el stride
                    row += strideP;
                    //Y reiniciar el col a 0
                    col = 0;
                }
            }
            return poolOutput;
        }
    }
}
