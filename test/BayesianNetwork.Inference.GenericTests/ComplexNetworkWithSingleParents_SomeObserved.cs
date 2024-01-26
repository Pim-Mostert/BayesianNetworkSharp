using BayesianNetwork.Inference.GenericTests.Helpers;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.GenericTests;

public class ComplexNetworkWithSingleParents_SomeObserved
{
    private Node _Q1, _Q2, _Q3, _Y1, _Y2, _Y3, _Y4, _Y5;
    private NaiveInferenceMachine _sut;
    private Evidence _evidence;

    [SetUp]
    public void Setup()
    {
        set_default_dtype(float64);

        _Q1 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2]), name: "Q1");
        _Q2 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 3]), parents: [_Q1], name: "Q2");
        _Q3 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([3, 2]), parents: [_Q2], name: "Q3");
        _Y1 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q1], isObserved: true, name: "Y1");
        _Y2 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([3, 3]), parents: [_Q2], isObserved: true, name: "Y2");
        _Y3 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([3, 4]), parents: [_Q2], isObserved: true, name: "Y3");
        _Y4 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q3], isObserved: true, name: "Y4");
        _Y5 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 3]), parents: [_Q3], isObserved: true, name: "Y5");

        BayesianNetwork bayesianNetwork = new([_Q1, _Q2, _Q3, _Y1, _Y2, _Y3, _Y4, _Y5]);

        _evidence = EvidenceBuilder.For(bayesianNetwork)
            .SetState(_Y1, new State([1, 0]))
            .SetState(_Y2, new State([0, 1, 0]))
            .SetState(_Y3, new State([1, 0, 1, 0]))
            .SetState(_Y4, new State([0, 1]))
            .SetState(_Y5, new State([1, 1, 0]))
            .Build();

        _sut = new NaiveInferenceMachine(bayesianNetwork);
        _sut.EnterEvidence(_evidence);
    }

    [Test]
    public void InferSingleNode_SomeObserved_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->i",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pQ1_expected /= pQ1_expected.sum();
        Tensor pQ2_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->j",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pQ2_expected /= pQ2_expected.sum();
        Tensor pQ3_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->k",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pQ3_expected /= pQ3_expected.sum();

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pQ3_actual = _sut.Infer(_Q3);

        // Assert
        Assert.Multiple(() =>
        {
            AssertHelpers.AssertTensorEqual(pQ1_expected, pQ1_actual);
            AssertHelpers.AssertTensorEqual(pQ2_expected, pQ2_actual);
            AssertHelpers.AssertTensorEqual(pQ3_expected, pQ3_actual);
        });
    }

    [Test]
    public void LogLikelihood_SomeObserved_Correct()
    {
        // Assign
        double expected = log(
            einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor()))
            .item<double>();

        // Act
        double actual = _sut.LogLikelihood;

        // Assert
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void InferSingleNodeWithParents_SomeObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->ij",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pQ1xQ2_expected /= pQ1xQ2_expected.sum();
        Tensor pQ2xQ3_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->jk",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pQ2xQ3_expected /= pQ2xQ3_expected.sum();
        Tensor pY1xQ1_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->ia",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pY1xQ1_expected /= pY1xQ1_expected.sum();
        Tensor pY2xQ2_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->jb",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pY2xQ2_expected /= pY2xQ2_expected.sum();
        Tensor pY3xQ2_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->jc",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pY3xQ2_expected /= pY3xQ2_expected.sum();
        Tensor pY4xQ3_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->kd",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pY4xQ3_expected /= pY4xQ3_expected.sum();
        Tensor pY5xQ3_expected = einsum("i, ij, jk, ia, jb, jc, kd, ke, a, b, c, d, e->ke",
                                           _Q1.Cpt, _Q2.Cpt, _Q3.Cpt, _Y1.Cpt, _Y2.Cpt, _Y3.Cpt, _Y4.Cpt, _Y5.Cpt,
                                           _evidence.GetState(_Y1).AsTensor(), _evidence.GetState(_Y2).AsTensor(), _evidence.GetState(_Y3).AsTensor(), _evidence.GetState(_Y4).AsTensor(), _evidence.GetState(_Y5).AsTensor());
        pY5xQ3_expected /= pY5xQ3_expected.sum();

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
            AssertHelpers.AssertTensorEqual(pQ1xQ2_expected, pQ2xQ1_actual);
            AssertHelpers.AssertTensorEqual(pQ2xQ3_expected, pQ3xQ2_actual);
            AssertHelpers.AssertTensorEqual(pY1xQ1_expected, pY1xQ1_actual);
            AssertHelpers.AssertTensorEqual(pY2xQ2_expected, pY2xQ2_actual);
            AssertHelpers.AssertTensorEqual(pY3xQ2_expected, pY3xQ2_actual);
            AssertHelpers.AssertTensorEqual(pY4xQ3_expected, pY4xQ3_actual);
            AssertHelpers.AssertTensorEqual(pY5xQ3_expected, pY5xQ3_actual);
        });
    }
}