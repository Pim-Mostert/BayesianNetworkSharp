using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class SingleParents_NoneObserved
{
    private Node _Q1;
    private Node _Q2;
    private Node _Y;

    private BayesianNetwork _bayesianNetwork;

    private IInferenceMachine _sut;

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
            .AddNode(_Q1)
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
            AssertTensorEqual(pQ1_expected, pQ1_actual);
            AssertTensorEqual(pQ2_expected, pQ2_actual);
            AssertTensorEqual(pY_expected, pY_actual);
        });
    }

    [Test]
    public void InferSingleNodeWithParents_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = torch.einsum("i, ij->ij", _Q1.Cpt, _Q2.Cpt);
        Tensor pQ2xY_expected = torch.einsum("i, ij, jk->jk", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            AssertTensorEqual(pQ1xQ2_expected, pQ1xQ2_actual);
            AssertTensorEqual(pQ2xY_expected, pQ2xY_actual);
        });
    }

    private static void AssertTensorEqual(Tensor expected, Tensor actual, double tolerance = 1e-5)
    {
        var expectedArray = expected.data<float>().ToArray();
        var actualArray = actual.data<float>().ToArray();

        Assert.That(expectedArray, Is.EqualTo(actualArray).Within(tolerance));
    }

    private static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size);

        return p / p.sum(dim: -1, keepdim: true);
    }
}