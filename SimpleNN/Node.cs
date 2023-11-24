using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace SimpleNN
{
    public class Node
    {
        public string Id { get; private set; }
        public float Bias { get; set; }
        public float TriggerValue { get; set; }
        public List<Neuron> InNeurons { get; set; }
        public List<Neuron> OutNeurons { get; set; }
        public float ActivatorValue => ActivatorFunctions.Activate(Activator, TriggerValue);
        public float ErrorValue { get; set; }
        public ActivatorType Activator { get; set; }

        public Node()
        {
            InNeurons = new List<Neuron>();
            OutNeurons = new List<Neuron>();
            Id = Random.Shared.Next().ToString("X");
        }

        public Node(ActivatorType activator) : this() => Activator = activator;

        public void UpdateTriggerValue()
        {
            TriggerValue = 0;
            foreach (Neuron n in InNeurons)
            {
                TriggerValue += n.Weight * n.Parent.ActivatorValue;
            }
            TriggerValue += Bias;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Node: {Id}");
            sb.AppendLine($" - Bias: {Bias:0.###}");
            sb.AppendLine($" - Activator: {Activator}");
            sb.AppendLine(" - Neurons In:");
            foreach (var n in InNeurons)
            {
                sb.AppendLine($"   - {n.Id}");
            }
            sb.AppendLine(" - Neurons Out:");
            foreach (var n in OutNeurons)
            {
                sb.AppendLine($"   - {n.Id}");
            }
            return sb.ToString();
        }
    }
}
