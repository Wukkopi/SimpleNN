using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    public enum ActivatorType
    {
        Linear = 0,
        Tanh = 1,
        DfTanh = 2,
        Sigmoid = 3,
        DfSigmoid = 4,
        Step20 = 5,
        Step50 = 6,
        Step80 = 7,
        ReLU = 8,
        SmoothReLU = 9,
        Clamp01 = 10,
    }
}
