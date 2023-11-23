using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SimpleNN
{
    public class Network
    {
        public List<List<Node>> nodes { get; set; }
        public List<Neuron> neurons { get; set; }

        public int LayerCount => nodes.Count;
        public int NodeCount => nodes.Sum(x => x.Count);
        public int NeuronCount => neurons.Count;

        public int InputCount => nodes[0].Count;
        public int OutputCount => nodes[LayerCount - 1].Count;

        private static JsonSerializerOptions serializerOptions = new JsonSerializerOptions { IncludeFields = true, ReferenceHandler = ReferenceHandler.Preserve };

        public Network()
        {
            nodes = new List<List<Node>>();
        }

        public Network(int layerCount) : this()
        {
            for (int i = 0; i < layerCount; i++)
            {
                nodes.Add(new List<Node>());
            }

            neurons = new List<Neuron>();
        }

        public Network(int[] layers, ActivatorType activator) : this(layers.Length)
        {
            for(int l = 0; l < LayerCount; l++)
            {
                for (int i = 0; i < layers[l]; i++)
                {
                    AddNode(l, activator);
                }
            }
        }

        public Neuron AddNeuron(Node child, Node parent)
        {
            Neuron neuron = new Neuron(parent, child);
            neurons.Add(neuron);
            parent.OutNeurons.Add(neuron);
            child.InNeurons.Add(neuron);
            return neuron;
        }
        public void RemoveNeuron(Neuron neuron)
        {
            neurons.Remove(neuron);
            neuron.Parent.OutNeurons.Remove(neuron);
            neuron.Child.InNeurons.Remove(neuron);
        }
        public Node AddNode(int layerIndex, ActivatorType activator, bool autowire = true)
        {
            Node newNode = new Node(activator);
            nodes[layerIndex].Add(newNode);
            
            if (!autowire)
            {
                return newNode;
            }

            if (layerIndex > 0)
            {
                // Add neurons TO this
                foreach (Node parent in nodes[layerIndex - 1])
                {
                    var neuron = new Neuron(parent, newNode);
                    newNode.InNeurons.Add(neuron);
                    parent.OutNeurons.Add(neuron);
                    neurons.Add(neuron);
                }
            }

            // Add neurons FROM this
            if (layerIndex < LayerCount - 1)
            {
                foreach (Node child in nodes[layerIndex + 1])
                {
                    var neuron = new Neuron(newNode, child);
                    newNode.OutNeurons.Add(neuron);
                    child.InNeurons.Add(neuron);
                    neurons.Add(neuron);
                }
            }
            return newNode;
        }

        public float[] Compute(float[] input)
        {
            if (input.Length != nodes[0].Count)
            {
                throw new Exception($"Wrong amount of inputs, expected {nodes[0].Count} while got only {input.Length}");
            }

            // set the input values
            for (int i = 0; i < nodes[0].Count; i++)
            {
                nodes[0][i].TriggerValue = input[i];
            }

            for (int i = 1; i < LayerCount; i++)
            {
                foreach(var node in nodes[i])
                {
                    node.UpdateTriggerValue();
                }
            }

            float[] result = new float[nodes[LayerCount - 1].Count];
            for(int i = 0; i < nodes[LayerCount - 1].Count; i++)
            {
                result[i] = nodes[LayerCount - 1][i].ActivatorValue;
            }

            return result;
        }

        public void Randomize()
        {
            foreach (List<Node> layer in nodes)
                foreach (Node node in layer)
                    node.Bias = Random.Shared.NextSingle() * 2f - 1f;

            foreach(Neuron neuron in neurons)
            {
                neuron.Weight = Random.Shared.NextSingle() * 2f - 1f;
            }
        }

        public void Mutate(float rate)
        {
            foreach (List<Node> layer in nodes)
            {
                foreach (Node node in layer)
                {
                    node.Bias += (Random.Shared.NextSingle() * 2f - 1f) * rate;
                }
            }
            foreach (Neuron neuron in neurons)
            {
                neuron.Weight += (Random.Shared.NextSingle() * 2f - 1f) * rate;
            }
        }

        public float BackPropagate(float[] expected, float rate)
        {
            float totalError = 0f;

            for (int i = LayerCount - 1; i > -1; i--)
            {
                for (int j = 0; j < nodes[i].Count; j++)
                {
                    Node node = nodes[i][j];
                    node.ErrorValue = node.ActivatorValue * ActivatorFunctions.Activate(ActivatorType.DfTanh, node.ActivatorValue) * rate;
                    if (i < LayerCount - 1)
                    {
                        List<Neuron> allNeurons = neurons.Where(n => n.Parent == node).ToList();
                        float errorMultiplier = 0f;
                        for(int k = 0; k < allNeurons.Count; k++)
                        {
                            errorMultiplier += allNeurons[k].Weight * allNeurons[k].Child.ErrorValue;
                            allNeurons[k].Weight += allNeurons[k].Child.ErrorValue * node.ActivatorValue;
                        }
                        node.ErrorValue *= errorMultiplier;
                    }
                    else
                    {
                        node.ErrorValue *= expected[j] - node.ActivatorValue;
                    }
                    node.Bias += node.ErrorValue;
                    totalError += Math.Abs(node.ErrorValue);
                }
            }
            return totalError;
        }
        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (var layer in nodes)
            {
                sb.AppendLine("============");
                foreach (var node in layer)
                {
                    
                }
            }

            return "";
        }

        public void SaveToStream(Stream stream)
        {
            stream.Position = 0;
            JsonSerializer.Serialize(stream, this, serializerOptions);
        }

        public static Network LoadFromStream(Stream stream)
        {
            stream.Position = 0;
            var network = JsonSerializer.Deserialize<Network>(stream, serializerOptions);
            if (network == null)
            {
                throw new Exception("Failed to read simple neural network from stream");
            }
            return network;
        }

        public static Network CreateCopyFrom(Network from)
        {
            MemoryStream temp = new MemoryStream();

            from.SaveToStream(temp);
            return LoadFromStream(temp);
        }
    }
}
