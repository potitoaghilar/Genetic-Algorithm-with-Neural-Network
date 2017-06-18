using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace Backpropagation
{
    public class NeuroNetwork : NeuralNetwork.NeuroNetwork
    {

        double learning_rate = .2;

        public NeuroNetwork(int input_nodes, int[] neurons_per_layer, int output_nodes, sbyte[] genome) : base(input_nodes, neurons_per_layer, output_nodes, genome) { }


        protected override void createPerceptrons() {
            perceptrons = new Perceptron[neurons_per_layer.Sum()];
        }

        public void trainNetwork(double[] input, double[] target, int ages) {

            for (int i = 0; i < ages; i++) {
                elaborateWithErrors(input, target);

                updateWeights();
            }

        }

        protected void elaborateWithErrors(double[] input, double[] target) {

            double[] output = elaborate(input).Reverse().ToArray();
            target = target.Reverse().ToArray();
            perceptrons = perceptrons.Reverse().ToArray();

            Perceptron[] perceptrons_in_last_layer = null, perceptrons_in_curr_layer = null;

            int neuron_n = 0, neuron_nn = 0;
            for (int i = hidden_layers_count - 1; i >= 0; i--) {

                perceptrons_in_curr_layer = new Perceptron[output_nodes];

                for (int o = 0; o < neurons_per_layer[i]; o++) {

                    if (i == hidden_layers_count - 1) {
                        
                        (perceptrons[neuron_n] as Perceptron).set_delta_error((target[o] - output[o]) * output[o] * (1 - output[o]));

                    } else {

                        (perceptrons[neuron_n] as Perceptron).set_delta_error(calc_partial_error(perceptrons_in_last_layer, neurons_per_layer[i] - o - 1) * output[o] * (1 - output[o]));

                    }

                    perceptrons_in_curr_layer[o] = perceptrons[neuron_n] as Perceptron;
                    neuron_n++;

                }

                perceptrons_in_last_layer = perceptrons_in_curr_layer;
                perceptrons_in_curr_layer = new Perceptron[neurons_per_layer[i]];

            }

        }

        private double calc_partial_error(Perceptron[] perceptrons, int weight_index) {

            double partial_error = 0;

            foreach (Perceptron p in perceptrons) {
                partial_error += p.get_delta_error() * p.getWeights()[weight_index];
            }

            return partial_error;

        }

        protected void updateWeights() {
            foreach (Perceptron p in perceptrons) {
                Console.WriteLine(p.get_delta_error());
            }
        }

    }
}
