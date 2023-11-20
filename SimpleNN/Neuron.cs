using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    [Serializable]
    public class Neuron
    {
        public float Weight { get; set; }
        public Node Parent { get; set; }
        public Node Child { get; set; }

        public Neuron(Node parent, Node child)
        {
            Parent = parent;
            Child = child;
        }
    }
}
