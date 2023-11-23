namespace SimpleNN.Tests
{
    public class NetworkTests
    {
        Network testNetwork;
        [SetUp]
        public void Setup()
        {
            //create a valid xor-nn
            testNetwork = new Network(3);

            testNetwork.AddNode(0, ActivatorType.ReLU);
            testNetwork.AddNode(0, ActivatorType.ReLU);

            Node node;

            node = testNetwork.AddNode(1, ActivatorType.ReLU);
            node.InNeurons[0].Weight = 1f;
            node.InNeurons[1].Weight = 1f;

            node = testNetwork.AddNode(1, ActivatorType.ReLU);
            node.InNeurons[0].Weight = 1f;
            node.InNeurons[1].Weight = 1f;
            node.Bias = -1f;

            node = testNetwork.AddNode(2, ActivatorType.ReLU);
            node.InNeurons[0].Weight = 1f;
            node.InNeurons[1].Weight = -2f;
        }

        [Test]
        public void TestObjectCountsInNetwork()
        {
            Assert.That(testNetwork.NodeCount, Is.EqualTo(5));
            Assert.That(testNetwork.NeuronCount, Is.EqualTo(6));
        }

        [Test]
        public void TestIO()
        {
            Assert.That(testNetwork.Compute(new float[] { 0, 0 }).Length, Is.EqualTo(1));
        }

        [TestCase(0, 0, 0)]
        [TestCase(1, 0, 1)]
        [TestCase(0, 1, 1)]
        [TestCase(1, 1, 0)]
        public void TestXor(float a, float b, float output)
        {
            Assert.That(testNetwork.Compute(new float[] { a, b })[0], Is.EqualTo(output));
        }

        [Test]
        public void TestSaveAndLoad()
        {
            using (var memory = new MemoryStream())
            {
                testNetwork.SaveToStream(memory);
                var n = Network.LoadFromStream(memory);

                Assert.That(n.NodeCount, Is.EqualTo(testNetwork.NodeCount));
                Assert.That(n.NeuronCount, Is.EqualTo(testNetwork.NeuronCount));
            }
        }
    }
}