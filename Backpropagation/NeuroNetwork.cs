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

        double learning_rate = .05;

        public NeuroNetwork(int input_nodes, int[] neurons_per_layer, int output_nodes, sbyte[] genome) : base(input_nodes, neurons_per_layer, output_nodes, genome) { }


        protected override void createPerceptrons() {
            perceptrons = new Perceptron[neurons_per_layer.Sum()];
        }

        protected override void setPerceptron(int index, sbyte[] weights, sbyte value) {
            perceptrons[index] = new Perceptron(weights, value);
        }

        public void trainNetwork(double[][] input_dataset, double[][] target_dataset, int ages) {

            for (int i = 0; i < ages; i++) {

                for (int o = 0; o < input_dataset.Length; o++)
                {
                    elaborateWithErrors(input_dataset[o], target_dataset[o]);
                    updateWeights(input_dataset[o]);
                }

            }

        }

        protected void elaborateWithErrors(double[] input, double[] target) {

            double[] output = elaborate(input).Reverse().ToArray();
            target = target.Reverse().ToArray();
            perceptrons = perceptrons.Reverse().ToArray();

            Perceptron[] perceptrons_in_last_layer = null, perceptrons_in_curr_layer = null;

            int neuron_n = 0;
            for (int i = hidden_layers_count - 1; i >= 0; i--) {

                perceptrons_in_curr_layer = new Perceptron[neurons_per_layer[i]];

                for (int o = 0; o < neurons_per_layer[i]; o++) {

                    if (i == hidden_layers_count - 1) {
                        
                        (perceptrons[neuron_n] as Perceptron).set_delta_error((target[o] - output[o]) * output[o] * (1 - output[o]));

                    } else {

                        (perceptrons[neuron_n] as Perceptron).set_delta_error(calc_partial_error(perceptrons_in_last_layer, neurons_per_layer[i] - o - 1) * perceptrons[neuron_n].getOutput() * (1 - perceptrons[neuron_n].getOutput()));

                    }

                    perceptrons_in_curr_layer[o] = perceptrons[neuron_n] as Perceptron;
                    neuron_n++;

                }

                perceptrons_in_last_layer = perceptrons_in_curr_layer;

            }

            perceptrons = perceptrons.Reverse().ToArray();

        }

        private double calc_partial_error(Perceptron[] perceptrons, int weight_index) {

            double partial_error = 0;

            foreach (Perceptron p in perceptrons) {
                partial_error += p.get_delta_error() * p.getWeights()[weight_index];
            }

            return partial_error;

        }

        protected void updateWeights(double[] input) {

            Perceptron[] perceptrons_in_curr_layer = null, perceptrons_in_previous_layer = null;

            int neuron_n = 0;
            for (int i = 0; i < hidden_layers_count; i++) {

                perceptrons_in_curr_layer = new Perceptron[neurons_per_layer[i]];

                for (int o = 0; o < neurons_per_layer[i]; o++) {

                    double[] weights = perceptrons[neuron_n].getWeights();
                    for (int w = 0; w < weights.Length; w++) {

                        if(perceptrons_in_previous_layer == null)
                            weights[w] += learning_rate * (perceptrons[neuron_n] as Perceptron).get_delta_error() * input[w];
                        else
                            weights[w] += learning_rate * (perceptrons[neuron_n] as Perceptron).get_delta_error() * perceptrons_in_previous_layer[w].getOutput();

                        perceptrons_in_curr_layer[o] = (perceptrons[neuron_n] as Perceptron);

                    }
                    perceptrons[neuron_n].setWeights(weights);

                    neuron_n++;

                }

                perceptrons_in_previous_layer = perceptrons_in_curr_layer;
                
            }
        }

    }
}
