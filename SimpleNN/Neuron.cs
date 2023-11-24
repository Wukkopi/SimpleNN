using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{

    public class Neuron : IDisposable
    {
        public string Id { get; private set; }
        public float Weight { get; set; }
        public Node Parent { get; set; }
        public Node Child { get; set; }

        public Neuron() => Id = Random.Shared.Next().ToString("X");
        public Neuron(Node parent, Node child, bool autowire = true) : this()
        {
            Parent = parent;
            Child = child;
            if (autowire)
            {
                Parent.OutNeurons.Add(this);
                Child.InNeurons.Add(this);
            }
        }

        public void Dispose()
        {
            Parent.OutNeurons.Remove(this);
            Child.InNeurons.Remove(this);
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Neuron: {Id}");
            sb.AppendLine($" - Weight: {Weight:0.###}");
            sb.AppendLine($" - Parent: {Parent.Id}");
            sb.AppendLine($" - Child: {Child.Id}");

            return sb.ToString();
        }
    }
}
