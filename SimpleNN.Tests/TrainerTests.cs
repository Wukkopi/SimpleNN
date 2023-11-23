using SimpleNNTrainer;

namespace SimpleNN.Tests
{
    public class TrainerTests
    {
        [Test]
        [Retry(3)]
        public void TestXor()
        {
            var testNetwork = new Network(new int[] { 2, 2, 1 }, ActivatorType.ReLU);

            var testIo = new List<float[]>
            {
                new float[] { 0, 0, 0 },
                new float[] { 1, 0, 1 },
                new float[] { 0, 1, 1 },
                new float[] { 1, 1, 0 }
            };

            var trainer = new MutateNetworkTrainer(testNetwork);
            trainer.AddTrainingData(testIo);
            var trainedNetwork = trainer.Train();

            Assert.That(trainedNetwork.Compute(new float[] { 0, 0 })[0], Is.LessThan(0.5f));
            Assert.That(trainedNetwork.Compute(new float[] { 0, 1 })[0], Is.GreaterThan(0.5f));
            Assert.That(trainedNetwork.Compute(new float[] { 1, 0 })[0], Is.GreaterThan(0.5f));
            Assert.That(trainedNetwork.Compute(new float[] { 1, 1 })[0], Is.LessThan(0.5f));
        }
    }
}
