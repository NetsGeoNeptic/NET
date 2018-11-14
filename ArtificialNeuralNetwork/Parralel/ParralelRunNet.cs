using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace ArtificialNeuralNetwork.Parralel
{
    class ParralelRunNet
    {
        private ManualResetEvent _doneEvent;
        private double[] _inputs;
        private Perzeptron _neuron;

        public ParralelRunNet(ManualResetEvent doneEvent, Perzeptron neuron, double[] actualInputs)
        {
            this._doneEvent = doneEvent;
            this._neuron = neuron;
            this._inputs = actualInputs;
        }

        public void ThreadPoolCallback(Object threadContext)
        {
            Calculate(_neuron, _inputs);
            _doneEvent.Set();
        }

        public void Calculate(Perzeptron current_neuron, double[] inputs)
        {
            double sum = 0;
            for (int i = 0; i < current_neuron.weighSignalLength; i++)
            {
                sum = sum + (inputs[i] * current_neuron.weighSignal[i]);
            }
            current_neuron.sum = sum;
            current_neuron.activation = f(current_neuron.sum - current_neuron.extraWeight);
        }

        // Sigmoid function
        public double f(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }
}
