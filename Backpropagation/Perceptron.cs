using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace Backpropagation
{
    public class Perceptron : NeuralNetwork.Perceptron
    {

        private double delta_error;

        public Perceptron(sbyte[] weights, sbyte bias) : base(weights, bias) { }

        public void set_delta_error(double delta_error)
        {
            this.delta_error = delta_error;
        }

        public double get_delta_error()
        {
            return delta_error;
        }

    }
}
