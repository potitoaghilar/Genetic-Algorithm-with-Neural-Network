using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GeneticAlgorithm;
using NeuralNetwork;
using Backpropagation;

namespace Agent
{
    public class Agent_NN
    {

        static Random random = new Random();

        static void Main(String[] args)
        {
            double[] input_vals = new double[] { .5, 1 };

            int input_length = input_vals.Length, output_legth = 2;
            int[] hiddenNeurons = new int[] { 3, 3 };

            sbyte[] genome = new sbyte[NeuralNetwork.NeuroNetwork.calculateGenomeLength(input_length, hiddenNeurons, output_legth)];
            for (int i = 0; i < 100; i++)
            {
                for (int g = 0; g < genome.Length; g++)
                    genome[g] = (sbyte)random.Next(-128, 127);

                NeuralNetwork.NeuroNetwork nn = new NeuralNetwork.NeuroNetwork(input_length, hiddenNeurons, output_legth, genome);
                Console.WriteLine(nn.elaborate(input_vals)[0]);
            }


            Console.ReadLine();
        }

    }

    public class Agent_GA
    {
        static void Main(String[] args)
        {
            double[] input_vals = new double[] { 0, 1, .1, .9, 0, 0, 0, .1, 1, 0, 0, 1, .1, .9, 0, 0, 0, .1, 1, 0, 0, 1, .1, .9, 0, 0, 0, .1, 1, 0, 0, 1, .1, .9, 0, 0, 0, .1, 1, 0, 0, 1, .1, .9, 0, 0, 0, .1, 1, 0 };

            int input_length = input_vals.Length, output_legth = 1;
            int[] hiddenNeurons = new int[] { 16, 16 };

            GeneticController gCtrl = new GeneticController(15, input_length, hiddenNeurons, output_legth);

            for (int i = 0; i < 100; i++)
            {
                Console.Clear();
                gCtrl.createGeneration();
                double[] fitness = new double[15];
                int index = 0;
                foreach (double[] o in gCtrl.executeGeneration(input_vals))
                {
                    if (o[0] < .6 && o[0] > .4)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        fitness[index] = 10 - (20 * Math.Abs(.5 - o[0]));
                    }
                    index++;
                    Console.WriteLine(o[0]);
                    Console.ForegroundColor = ConsoleColor.White;
                }
                gCtrl.setFitness(fitness);
            }
            Console.ReadLine();

        }

    }
    
    public class Agent_BACKPROPAGATION
    {

        static Random random = new Random();

        static void Main(String[] args)
        {
            double[] input = new double[] { 0, 1 }, target = new double[] { 1, 0 };

            int input_length = input.Length, output_legth = target.Length;
            int[] hiddenNeurons = new int[] { 3 };

            sbyte[] genome = new sbyte[NeuralNetwork.NeuroNetwork.calculateGenomeLength(input_length, hiddenNeurons, output_legth)];
            
            
            

            Console.ReadLine();
        }

    }
}
