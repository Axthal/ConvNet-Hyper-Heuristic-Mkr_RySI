using System;
using System.Collections.Generic;
using System.Text;

namespace CNN_RySI.MLP.Components
{
    public class Neuron
    {
        /// <summary>
        /// Array con las entradas de la neurona (Dendritas)
        /// </summary>
        public Connection[] Inputs { get; set; }
        /// <summary>
        /// Salida de la neurona (Axón)
        /// </summary>
        public double Output { get; set; }
        /// <summary>
        /// Atributo que contiene el método con la Regla de propagación. El método debe recibir un arreglo de Conexiones y debe devolver el resultado en double.
        /// </summary>
        public Func<Connection[], double> PropagationRule { get; set; }
        /// <summary>
        /// Atributo que contiene el método con la Función de Activación. El método debe recibir un número double y debe devolver el resultado en double.
        /// </summary>
        public Func<double, double> ActivationFunction { get; set; }
        /// <summary>
        /// Atributo que contiene el método con la Función de Salida. El método debe recibir un número double y debe devolver el resultado en double.
        /// </summary>
        public Func<double, double> OutputFunction { get; set; }
        public Neuron()
        {
            Inputs = new Connection[] { new Connection() };
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="PropagationRule">Método que determinará la regla de propagación de la neurona. Ver algunos métodos en la clase -PropagationRule-</param>
        /// <param name="ActivationFunction">Método que determinará la función de activación de la neurona. Ver algunos métodos en la clase -ActivationFunction-</param>
        /// <param name="OutputFunction">Método que determinará la función de salida de la neurona. Ver algunos métodos en la clase -OutputFunction-</param>
        public Neuron(
            Func<Connection[], double> PropagationRule,
            Func<double, double> ActivationFunction,
            Func<double, double> OutputFunction)
        {
            this.PropagationRule = PropagationRule;
            this.ActivationFunction = ActivationFunction;
            this.OutputFunction = OutputFunction;
            Inputs = new Connection[] { new Connection() };
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="PropagationRule">Método que determinará la regla de propagación de la neurona. Ver algunos métodos en la clase -PropagationRule-</param>
        /// <param name="ActivationFunction">Método que determinará la función de activación de la neurona. Ver algunos métodos en la clase -ActivationFunction-</param>
        /// <param name="OutputFunction">Método que determinará la función de salida de la neurona. Ver algunos métodos en la clase -OutputFunction-</param>
        /// <param name="TotalInputs">Total de conexiones de entrada que tendrá la neurona</param>
        public Neuron(
            Func<Connection[], double> PropagationRule,
            Func<double, double> ActivationFunction,
            Func<double, double> OutputFunction,
            int TotalInputs)
        {
            this.PropagationRule = PropagationRule;
            this.ActivationFunction = ActivationFunction;
            this.OutputFunction = OutputFunction;
            Inputs = new Connection[TotalInputs];
            for (int i = 0; i < TotalInputs; i++)
            {
                Inputs[i] = new Connection();
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="PropagationRule"></param>
        /// <param name="ActivationFunction"></param>
        /// <param name="OutputFunction"></param>
        /// <param name="Inputs"></param>
        public Neuron(
            Func<Connection[], double> PropagationRule,
            Func<double, double> ActivationFunction,
            Func<double, double> OutputFunction,
            Connection[] Inputs)
        {
            this.PropagationRule = PropagationRule;
            this.ActivationFunction = ActivationFunction;
            this.OutputFunction = OutputFunction;
            this.Inputs = new Connection[Inputs.Length];
            Array.Copy(Inputs,this.Inputs, Inputs.Length);
        }
        /// <summary>
        /// Verifica que los atributos con los métodos no sean nulos. En caso contrario, manda una excepción.
        /// </summary>
        private void CheckMethods()
        {
            if (PropagationRule == null)
                throw new System.ArgumentException("Propagation Rule not implemented");
            if (ActivationFunction == null)
                throw new System.ArgumentException("Activation Function not implemented");
            if (OutputFunction == null)
                throw new System.ArgumentException("Output Function not implemented");
        }
        /// <summary>
        /// Método que activa la neurona para su procesamiento
        /// </summary>
        /// <returns>El resultado del procesamiento</returns>
        public double Activation()
        {
            CheckMethods();
            try
            {
                Output = PropagationRule(Inputs);
                Output = ActivationFunction(Output);
                Output = OutputFunction(Output);
                return Output;
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }
    }
}
