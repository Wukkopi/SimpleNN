using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    public static class ActivatorFunctions
    {
        public static float Linear(float accum) => accum;
        public static float Tanh(float accum) => (float)Math.Tanh(accum);
        public static float DfTanh(float accum) => (float)(1f - Math.Pow(Math.Tanh(accum), 2));
        public static float Sigmoid(float accum) => (float)(1f / (1f - Math.Pow(Math.E, accum)));
        public static float dfSigmoid(float accum) => accum * (1f - accum);
        public static float Step20(float accum) => accum >= 0.2f ? 1f : 0f;
        public static float Step50(float accum) => accum >= 0.5f ? 1f : 0f;
        public static float Step80(float accum) => accum >= 0.8f ? 1f : 0f;
        public static float ReLU(float accum) => Math.Max(0, accum);
        public static float SmoothReLU(float accum) => (float)Math.Log(1 + Math.Pow(Math.E, accum));
        public static float Clamp01(float accum) => accum < 0f ? 0f : accum > 1f ? 1f : accum;
    }
}
