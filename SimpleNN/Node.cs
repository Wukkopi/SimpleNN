using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;

namespace SimpleNN
{
    [Serializable]
    public class Node
    {
        public string Id { get; private set; }
        public float Bias { get; set; }
        public Func<float, float> Activator { get; private set; }
        public float TriggerValue { get; set; }
        public float ActivatorValue => Activator(TriggerValue);
        public float ErrorValue { get; set; }
        public List<Neuron> InNeurons { get; set; }
        public List<Neuron> OutNeurons { get; set; }

        public Node(Func<float, float> activator)
        {
            Activator = activator;
            InNeurons = new List<Neuron>();
            OutNeurons = new List<Neuron>();
            Id = Random.Shared.Next().ToString("X");
        }

        public void UpdateTriggerValue()
        {
            TriggerValue = 0;
            foreach (Neuron n in InNeurons)
            {
                TriggerValue += n.Weight * n.Parent.ActivatorValue;
            }
            TriggerValue += Bias;
        }
    }
}
