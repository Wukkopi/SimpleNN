using System;
using System.Collections.Generic;

namespace SimpleNN
{
    [Serializable]
    public class Node
    {
        public float Bias { get; set; }
        public Func<float, float> Activator { get; private set; }
        public float TriggerValue { get; set; }
        public float ActivatorValue => Activator(TriggerValue);
        public float ErrorValue { get; set; }

        public Node(Func<float, float> activator)
        {
            Activator = activator;
        }

        public void UpdateTriggerValue(List<Neuron> neurons)
        {
            TriggerValue = 0;
            foreach (Neuron n in neurons)
            {
                TriggerValue += n.Weight * n.Parent.ActivatorValue;
            }
            TriggerValue += Bias;
        }
    }
}
