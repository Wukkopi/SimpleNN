namespace SimpleNN.Tests
{
    public class Tests
    {
        Network testNetwork;
        [SetUp]
        public void Setup()
        {
            //create a known xor-nn
            testNetwork = new Network(3);
            
            testNetwork.AddNode(0, ActivatorFunctions.Linear);
            testNetwork.AddNode(0, ActivatorFunctions.Linear);

            Node node;

            node = testNetwork.AddNode(1, ActivatorFunctions.Linear);
            node.InNeurons[0].Weight = 0.38f;
            node.InNeurons[1].Weight = 1.13f;
            node.Bias = 1.18f;

            node = testNetwork.AddNode(1, ActivatorFunctions.Linear);
            node.InNeurons[0].Weight = -1.08f;
            node.InNeurons[1].Weight = -0.56f;
            node.Bias = -0.33f;

            node = testNetwork.AddNode(2, ActivatorFunctions.Linear);
            node.InNeurons[0].Weight = -1.49f;
            node.InNeurons[1].Weight = -0.46f;
            node.Bias = 1.23f;
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
    }
}