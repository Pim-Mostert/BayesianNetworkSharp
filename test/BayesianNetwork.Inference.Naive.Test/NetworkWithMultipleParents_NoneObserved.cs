using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NetworkWithMultipleParents_NoneObserved
{
    private Node _Q1, _Q2, _Y;
    private NaiveInferenceMachine _sut;

    [SetUp]
    public void Setup()
    {
        _Q1 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2]), name: "Q1");
        _Q2 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q1], name: "Q2");
        _Y = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 2, 2]), parents: [_Q1, _Q2], name: "Y");

        BayesianNetwork bayesianNetwork = new(nodes: [_Q1, _Q2, _Y]);

        _sut = new NaiveInferenceMachine(bayesianNetwork);
    }

    [Test]
    public void InferSingleNode_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = torch.einsum("i->i", _Q1.Cpt);
        Tensor pQ2_expected = torch.einsum("i, ij->j", _Q1.Cpt, _Q2.Cpt);
        Tensor pY_expected = torch.einsum("i, ij, ijk->k", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

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
        Tensor pQ1xpQ2xY_expected = torch.einsum("i, ij, ijk->ijk", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ1xpQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            Helpers.AssertTensorEqual(pQ1xQ2_expected, pQ1xQ2_actual);
            Helpers.AssertTensorEqual(pQ1xpQ2xY_expected, pQ1xpQ2xY_actual);
        });
    }
}