using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    public static class ActivatorFunctions
    {
        private static float Linear(float accum) => accum;
        private static float Tanh(float accum) => (float)Math.Tanh(accum);
        private static float DfTanh(float accum) => (float)(1f - Math.Pow(Math.Tanh(accum), 2));
        private static float Sigmoid(float accum) => (float)(1f / (1f - Math.Pow(Math.E, accum)));
        private static float DfSigmoid(float accum) => accum * (1f - accum);
        private static float Step20(float accum) => accum >= 0.2f ? 1f : 0f;
        private static float Step50(float accum) => accum >= 0.5f ? 1f : 0f;
        private static float Step80(float accum) => accum >= 0.8f ? 1f : 0f;
        private static float ReLU(float accum) => Math.Max(0, accum);
        private static float SmoothReLU(float accum) => (float)Math.Log(1 + Math.Pow(Math.E, accum));
        private static float Clamp01(float accum) => accum < 0f ? 0f : accum > 1f ? 1f : accum;

        public static float Activate(ActivatorType activator, float accum)
        {
            return activator switch
            {
                ActivatorType.Linear => Linear(accum),
                ActivatorType.Tanh => Tanh(accum),
                ActivatorType.DfTanh => DfTanh(accum),
                ActivatorType.Sigmoid => Sigmoid(accum),
                ActivatorType.DfSigmoid => DfSigmoid(accum),
                ActivatorType.Step20 => Step20(accum),
                ActivatorType.Step50 => Step50(accum),
                ActivatorType.Step80 => Step80(accum),
                ActivatorType.ReLU => ReLU(accum),
                ActivatorType.SmoothReLU => SmoothReLU(accum),
                ActivatorType.Clamp01 => Clamp01(accum),
                _ => throw new NotImplementedException("Unknown activator called")
            };
        }

    }
}
