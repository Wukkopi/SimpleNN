using SimpleNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNNTrainer
{
    public abstract class NetworkTrainer
    {
        private readonly float trainTestSplit;
        private int testDataSplitIndex => (int)(trainData.Count * trainTestSplit);
        private readonly List<float[]> trainData = new List<float[]>();
        public List<float[]> GetTrainSet() => trainData.GetRange(0, testDataSplitIndex);
        public List<float[]> GetTestSet() => trainData.GetRange(testDataSplitIndex, trainData.Count - testDataSplitIndex);

        protected NetworkTrainer(float split) => trainTestSplit = split;

        public virtual Network Train() => throw new NotImplementedException();
        public void AddTrainingData(List<float[]> trainingData) => trainData.AddRange(trainingData);
    }
}
