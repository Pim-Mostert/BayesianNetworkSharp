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
            Cpt = Helpers.GenerateRandomProbabilityMatrix([2]),
            Name = "Q1"
        };
        _Q2 = new Node
        {
            Cpt = Helpers.GenerateRandomProbabilityMatrix([2, 2]),
            Name = "Q2"
        };
        _Y = new Node
        {
            Cpt = Helpers.GenerateRandomProbabilityMatrix([2, 2]),
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
            Helpers.AssertTensorEqual(pQ1_expected, pQ1_actual);
            Helpers.AssertTensorEqual(pQ2_expected, pQ2_actual);
            Helpers.AssertTensorEqual(pY_expected, pY_actual);
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
            Helpers.AssertTensorEqual(pQ1xQ2_expected, pQ1xQ2_actual);
            Helpers.AssertTensorEqual(pQ2xY_expected, pQ2xY_actual);
        });
    }
}