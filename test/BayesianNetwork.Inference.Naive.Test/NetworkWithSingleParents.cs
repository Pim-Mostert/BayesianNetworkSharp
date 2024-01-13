using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NetworkWithSingleParents
{
    private Node _Q1 { get; set; }
    private Node _Q2 { get; set; }
    private Node _Y { get; set; }

    private BayesianNetwork _bayesianNetwork { get; set; }

    private IInferenceMachine _sut { get; set; }

    [SetUp]
    public void Setup()
    {
        _Q1 = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2]),
            Name = "Q1"
        };
        _Q2 = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2, 2]),
            Name = "Q2"
        };
        _Y = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2, 2]),
            Name = "Y"
        };

        _bayesianNetwork = new BayesianNetworkBuilder()
            .AddRootNode(_Q1)
            .AddNode(_Q2, parent: _Q1)
            .AddNode(_Y, parent: _Q2)
            .Build();

        _sut = new NaiveInferenceMachine(_bayesianNetwork);
    }

    [Test]
    public void InferSingleNode_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = torch.einsum("i->i", _Q1.Cpt);
        Tensor pQ2_expected = torch.einsum("i, ij->j", _Q1.Cpt, _Q2.Cpt);
        Tensor pY_expected = torch.einsum("i, ij, jk->k", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pY_actual = _sut.Infer(_Y);

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(pQ1_actual, Is.EqualTo(pQ1_expected).Within(1e-5));
            Assert.That(pQ2_actual, Is.EqualTo(pQ2_expected).Within(1e-5));
            Assert.That(pY_actual, Is.EqualTo(pY_expected).Within(1e-5));
        });
    }

    private static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size);

        return p / p.sum(dim: -1, keepdim: true);
    }
}