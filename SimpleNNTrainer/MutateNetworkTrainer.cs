using Microsoft.VisualBasic;
using SimpleNN;
using System;
using System.Net.Sockets;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json.Nodes;

namespace SimpleNNTrainer
{
    public class MutateNetworkTrainer : NetworkTrainer
    {
        public struct Options
        {
            public int Generations = 10000;
            public float MutationRate = 0.1f;
            public int PopulationPerGeneration = 10;
  
            public Options() { }
        }
        private readonly Options options;
        private readonly List<Network> population = new List<Network>();

        public MutateNetworkTrainer(Network network) : this(network, new Options()) { }

        public MutateNetworkTrainer(Network network, Options options) : base(0f)
        {
            this.options = options;

            for(var i = 0; i < options.PopulationPerGeneration; i++)
            {
                population.Add(Network.CreateCopyFrom(network));
            }
        }

        public override Network Train()
        {
            var taskFactory = new TaskFactory();
            int generation = 0;
            List<float[]> testSet = GetTestSet();
            List<Task<float>> tasks = new List<Task<float>>();
            Network bestNetwork = null;

            while (generation < options.Generations)
            {
                generation++;
                tasks.Clear();
                foreach (var n in population)
                {
                    Task<float> t = taskFactory.StartNew(() =>
                    {
                        n.Mutate(options.MutationRate);
                        var totalError = 0f;
                        foreach (var testIo in testSet)
                        {
                            var result = n.Compute(testIo[0 .. n.InputCount]);
                            for (var i = 0; i < result.Length; i++)
                            {
                                totalError += Math.Abs(result[i] - testIo[i + n.InputCount]);
                            }
                        }
                        return totalError;
                    });
                    tasks.Add(t);
                }

                Task.WaitAll(tasks.ToArray());
                
                bestNetwork = population.ElementAt(tasks.FindIndex(t => t.Result == tasks.Min(x => x.Result)));

                for (var i = 0; i < options.PopulationPerGeneration; i++)
                {
                    population[i] = Network.CreateCopyFrom(bestNetwork);
                }
            }
            return bestNetwork;
        }
    }
}