using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class ComplexNetworkWithSingleParents_NoneObserved
{
    private Node _Q1, _Q2, _Q3, _Y1, _Y2, _Y3, _Y4, _Y5;
    private NaiveInferenceMachine _sut;

    [SetUp]
    public void Setup()
    {
        torch.set_default_dtype(torch.float64);

        _Q1 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2]), name: "Q1");
        _Q2 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 3]), parents: [_Q1], name: "Q2");
        _Q3 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([3, 2]), parents: [_Q2], name: "Q3");
        _Y1 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q1], name: "Y1");
        _Y2 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([3, 3]), parents: [_Q2], name: "Y2");
        _Y3 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([3, 4]), parents: [_Q2], name: "Y3");
        _Y4 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q3], name: "Y4");
        _Y5 = new Node(cpt: Helpers.GenerateRandomProbabilityMatrix([2, 3]), parents: [_Q3], name: "Y5");

        BayesianNetwork bayesianNetwork = new([_Q1, _Q2, _Q3, _Y1, _Y2, _Y3, _Y4, _Y5]);

        _sut = new NaiveInferenceMachine(bayesianNetwork);
    }

    [Test]
    public void InferSingleNode_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->i",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pQ2_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->j",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pQ3_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->k",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pQ3_actual = _sut.Infer(_Q3);

        // Assert
        Assert.Multiple(() =>
        {
            Helpers.AssertTensorEqual(pQ1_expected, pQ1_actual);
            Helpers.AssertTensorEqual(pQ2_expected, pQ2_actual);
            Helpers.AssertTensorEqual(pQ3_expected, pQ3_actual);
        });
    }

    [Test]
    public void InferSingleNodeWithParents_NoObservations_CorrectInference()
    {
        // Assign
        // Assign
        Tensor pQ2xQ1_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->ij",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pQ3xQ2_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->jk",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pY1xQ1_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->ia",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pY2xQ2_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->jb",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pY3xQ2_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->jc",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pY4xQ3_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->kd",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);
        Tensor pY5xQ3_expected = torch.einsum("i, ij, jk, ia, jb, jc, kd, ke->ke",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt);

        // Act
        Tensor pQ2xQ1_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ3xQ2_actual = _sut.Infer(_Q3, includeParents: true);
        Tensor pY1xQ1_actual = _sut.Infer(_Y1, includeParents: true);
        Tensor pY2xQ2_actual = _sut.Infer(_Y2, includeParents: true);
        Tensor pY3xQ2_actual = _sut.Infer(_Y3, includeParents: true);
        Tensor pY4xQ3_actual = _sut.Infer(_Y4, includeParents: true);
        Tensor pY5xQ3_actual = _sut.Infer(_Y5, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            Helpers.AssertTensorEqual(pQ2xQ1_expected, pQ2xQ1_actual);
            Helpers.AssertTensorEqual(pQ3xQ2_expected, pQ3xQ2_actual);
            Helpers.AssertTensorEqual(pY1xQ1_expected, pY1xQ1_actual);
            Helpers.AssertTensorEqual(pY2xQ2_expected, pY2xQ2_actual);
            Helpers.AssertTensorEqual(pY3xQ2_expected, pY3xQ2_actual);
            Helpers.AssertTensorEqual(pY4xQ3_expected, pY4xQ3_actual);
            Helpers.AssertTensorEqual(pY5xQ3_expected, pY5xQ3_actual);
        });
    }
}