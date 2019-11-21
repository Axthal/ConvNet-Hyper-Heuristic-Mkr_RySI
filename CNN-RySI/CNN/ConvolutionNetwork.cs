using CNN_RySI.CNN.Components;
using CNN_RySI.Tools;
using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.CNN
{
    public class ConvolutionNetwork
    {
        private ConvLayer[] Net_Layers;
        private string Architecture;
        private int ImgSizeIn;
        private int TotalValuesKernel;
        private int ImgSizeOut;
        /// <summary>
        /// Función que contruye el modelo convolucional
        /// </summary>
        /// <param name="Layers">Vector de objetos tipo 'Capa Convolutiva'</param>
        /// <param name="imgSizeIn">Tamaño de la imagen de entrada para las que se entrenará el modelo (ancho o alto)</param>
        /// <param name="imgDimensionIn">Dimensión de la imagen de entrada para las que se entrenará el modelo</param>
        /// <param name="imgSizeOut">Tamaño del mapa de características que producirá el modelo (ancho o alto)</param>
        /// <param name="imgDimensionOut">Dimensión del mapa de características que producirá el modelo</param>
        /// <returns>Total de valores de kernel que tendrá el modelo</returns>
        public int Build(ConvLayer[] Layers, int imgSizeIn, int imgDimensionIn, out int imgSizeOut, out int imgDimensionOut)
        {
            //Obtener las capas que contendrá este modelo y copiarlas de forma interna
            Net_Layers = new ConvLayer[Layers.Length];
            //Array.Copy(Layers, Net_Layers, Net_Layers.Length);
            Layers.CopyTo(Net_Layers,0);
            //Obtener el tamaño estándar de la imagen (ej. 28 x 28 píxeles, sería el 28).
            ImgSizeIn = imgSizeIn;
            //Obtener el número total de valores de kernel que serán entrenados
            TotalValuesKernel = ValidateNetwork(imgSizeIn, imgDimensionIn, out imgSizeOut, out imgDimensionOut);
            //Obtener el tamaño que tendrá la imagen al finalizar la ejecución del modelo
            ImgSizeOut = imgSizeOut;
            //Retornar el total de valores de kernel
            return TotalValuesKernel;
        }
        public ConvLayer[] GetLayers()
        {
            return Net_Layers;
        }
        /// <summary>
        /// Método que validará la configuración del modelo, construyéndolo mientras lo hace, obteniendo valores importantes a ser usados
        /// </summary>
        /// <param name="imgSizeIn">Tamaño de la imagen de entrada para las que se entrenará el modelo (ancho o alto)</param>
        /// <param name="imgDimensionIn">Dimensión de la imagen de entrada para las que se entrenará el modelo</param>
        /// <param name="imgSizeOut">Tamaño del mapa de características que producirá el modelo (ancho o alto)</param>
        /// <param name="imgDimensionOut">Dimensión del mapa de características que producirá el modelo</param>
        /// <returns></returns>
        private int ValidateNetwork(int imgSizeIn, int imgDimensionIn, out int imgSizeOut, out int imgDimensionOut)
        {
            bool valid;
            Architecture = "";
            int totalKernelValues = 0;
            int ISI = imgSizeIn;
            int IDI = imgDimensionIn;
            int ISO = 0, IDO = 0;
            //Por cada una de las capas del modelo
            for (int ixCNN = 0; ixCNN < Net_Layers.Length; ixCNN++)
            {
                //Obtener la veracidad de cada una de las capas, construyéndolas en caso de ser válidas
                valid = Net_Layers[ixCNN].CanProcessImage(ISI, IDI, out ISO, out IDO);
                if (!valid)
                    throw new Exception($"No podrá pasar de la capa {ixCNN}, intenta con otros parámetros (tamaño de kernel, stride o padding)");
                //Obtener la arquitectura del modelo a partir de cada una de las capas
                Architecture += Net_Layers[ixCNN].GetLayerArchitecture() + " | ";
                //Obtener el total de valores a entrenar del modelo a partir de cada una de las capas
                totalKernelValues += Net_Layers[ixCNN].GetTotalValuesLayer();
                //El output de la capa actual será el input de la siguiente, incluyendo el tamaño y dimensión
                ISI = ISO;
                IDI = IDO;
            }
            //Retornar valores
            imgSizeOut = ISO;
            imgDimensionOut = IDO;
            return totalKernelValues;
        }
        /// <summary>
        /// Método que coloca los pesos convolutivos dentro del modelo y procesa la imagen ingresada
        /// </summary>
        /// <param name="totalValues">Vector de pesos convolutivos</param>
        /// <param name="image">Arreglo multidimensional tipo jagged de la imagen</param>
        /// <returns></returns>
        public double[] FillAndProcess(double[] totalValues, double[][][] image)
        {
            if (image[0].Length != ImgSizeIn)
                throw new Exception($"Las dimensiones de la imagen que se ingresó ({image[0].Length}) no corresponden con las que se validó ({ImgSizeIn})");
            if (totalValues.Length != TotalValuesKernel)
                throw new Exception($"La longitud de los valores ingresados ({totalValues.Length}) es diferente a la registrada en el modelo ({TotalValuesKernel})");
            int ixAK = 0;
            int ixZK;
            //Por cada una de las capas del modelo
            for (int ixCNN = 0; ixCNN < Net_Layers.Length; ixCNN++)
            {
                //Obtener el tamaño de valores de kernel de la capa
                ixZK = Net_Layers[ixCNN].GetTotalValuesLayer();
                //Crear un vector temporal de igual tamaño
                double[] temp = new double[ixZK];
                //Copiar la sección correspondiente en el vector temporal
                Array.Copy(totalValues, ixAK, temp, 0, ixZK);
                //Llenar la capa con los valores de kernel obtenidos
                Net_Layers[ixCNN].FillAllKernels(temp);
                //Aumentar el index para la siguiente capa
                ixAK += ixZK;
                //Aplicar la capa de la sección de convolución correspondiente
                image = Net_Layers[ixCNN].ApplyLayer(image);
            }
            return DataHelper.GetVector_FromJaggedArray(image);
        }
        /// <summary>
        /// Método que coloca los pesos convolutivos dentro del modelo
        /// </summary>
        /// <param name="totalValues">Vector de pesos convolutivos</param>
        public void FillWeights(double[] totalValues)
        {
            int ixAK = 0;
            int ixZK;
            //Por cada una de las capas del modelo
            for (int ixCNN = 0; ixCNN < Net_Layers.Length; ixCNN++)
            {
                //Obtener el tamaño de valores de kernel de la capa
                ixZK = Net_Layers[ixCNN].GetTotalValuesLayer();
                //Crear un vector temporal de igual tamaño
                double[] temp = new double[ixZK];
                //Copiar la sección correspondiente en el vector temporal
                Array.Copy(totalValues, ixAK, temp, 0, ixZK);
                //Llenar la capa con los valores de kernel obtenidos
                Net_Layers[ixCNN].FillAllKernels(temp);
                //Aumentar el index para la siguiente capa
                ixAK += ixZK;
            }
        }
        /// <summary>
        /// Método que procesa la imagen ingresada
        /// </summary>
        /// <param name="image">Arreglo multidimensional tipo jagged de la imagen</param>
        /// <returns></returns>
        public double[] ProcessImage(double[][][] image)
        {
            if (image[0].Length != ImgSizeIn)
                throw new Exception($"Las dimensiones de la imagen que se ingresó ({image[0].Length}) no corresponden con las que se validó ({ImgSizeIn})");
            //Por cada una de las capas del modelo
            for (int ixCNN = 0; ixCNN < Net_Layers.Length; ixCNN++)
                //Aplicar la capa de la sección de convolución correspondiente
                image = Net_Layers[ixCNN].ApplyLayer(image);
            return DataHelper.GetVector_FromJaggedArray(image);
        }
        /// <summary>
        /// Método que retorna un string con la descripción de arquitectura del modelo convolucional
        /// </summary>
        /// <returns></returns>
        public string GetArchitecture()
        {
            return Architecture;
        }
    }
}
