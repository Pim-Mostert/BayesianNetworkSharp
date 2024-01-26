using BayesianNetwork.Inference.GenericTests.Helpers;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.GenericTests;

public class NetworkWithSingleParents_NoneObserved
{
    private Node _Q1, _Q2, _Y;
    private NaiveInferenceMachine _sut;

    [SetUp]
    public void Setup()
    {
        set_default_dtype(float64);

        _Q1 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2]), name: "Q1");
        _Q2 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q1], name: "Q2");
        _Y = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q2], name: "Y");

        BayesianNetwork bayesianNetwork = new(nodes: [_Q1, _Q2, _Y]);

        _sut = new NaiveInferenceMachine(bayesianNetwork);
    }

    [Test]
    public void InferSingleNode_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = einsum("i->i", _Q1.Cpt);
        Tensor pQ2_expected = einsum("i, ij->j", _Q1.Cpt, _Q2.Cpt);
        Tensor pY_expected = einsum("i, ij, jk->k", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pY_actual = _sut.Infer(_Y);

        // Assert
        Assert.Multiple(() =>
        {
            AssertHelpers.AssertTensorEqual(pQ1_expected, pQ1_actual);
            AssertHelpers.AssertTensorEqual(pQ2_expected, pQ2_actual);
            AssertHelpers.AssertTensorEqual(pY_expected, pY_actual);
        });
    }

    [Test]
    public void InferSingleNodeWithParents_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = einsum("i, ij->ij", _Q1.Cpt, _Q2.Cpt);
        Tensor pQ2xY_expected = einsum("i, ij, jk->jk", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            AssertHelpers.AssertTensorEqual(pQ1xQ2_expected, pQ1xQ2_actual);
            AssertHelpers.AssertTensorEqual(pQ2xY_expected, pQ2xY_actual);
        });
    }
}