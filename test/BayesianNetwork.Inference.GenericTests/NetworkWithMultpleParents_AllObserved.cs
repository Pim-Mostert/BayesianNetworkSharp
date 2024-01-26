using BayesianNetwork.Inference.GenericTests.Helpers;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.GenericTests;

public class NetworkWithMultpleParents_AllObserved
{
    private Node _Q1, _Q2, _Y;
    private NaiveInferenceMachine _sut;
    private Evidence _evidence;

    [SetUp]
    public void Setup()
    {
        set_default_dtype(float64);

        _Q1 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2]), name: "Q1", isObserved: true);
        _Q2 = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2]), parents: [_Q1], name: "Q2", isObserved: true);
        _Y = new Node(cpt: TensorHelpers.GenerateRandomProbabilityMatrix([2, 2, 2]), parents: [_Q1, _Q2], name: "Y", isObserved: true);

        BayesianNetwork bayesianNetwork = new(nodes: [_Q1, _Q2, _Y]);

        _evidence = EvidenceBuilder.For(bayesianNetwork)
            .SetState(_Q1, new State([1, 0]))
            .SetState(_Q2, new State([0, 1]))
            .SetState(_Y, new State([1, 0]))
            .Build();

        _sut = new NaiveInferenceMachine(bayesianNetwork);
        _sut.EnterEvidence(_evidence);
    }

    [Test]
    public void InferSingleNode_AllObserved_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = einsum("i, ij, ijk, i, j, k->i",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Q1).AsTensor(),
            _evidence.GetState(_Q2).AsTensor(),
            _evidence.GetState(_Y).AsTensor());
        pQ1_expected /= pQ1_expected.sum();
        Tensor pQ2_expected = einsum("i, ij, ijk, i, j, k->j",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Q1).AsTensor(),
            _evidence.GetState(_Q2).AsTensor(),
            _evidence.GetState(_Y).AsTensor());
        pQ2_expected /= pQ2_expected.sum();
        Tensor pY_expected = einsum("i, ij, ijk, i, j, k->k",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Q1).AsTensor(),
            _evidence.GetState(_Q2).AsTensor(),
            _evidence.GetState(_Y).AsTensor());
        pY_expected /= pY_expected.sum();

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pY_actual = _sut.Infer(_Y);

        // Assert
        Assert.Multiple(() =>
        {
            AssertHelpers.AssertTensorEqual(pQ1_actual, pQ1_expected);
            AssertHelpers.AssertTensorEqual(pQ2_actual, pQ2_expected);
            AssertHelpers.AssertTensorEqual(pY_actual, pY_expected);
        });
    }

    [Test]
    public void LogLikelihood_AllObserved_Correct()
    {
        // Assign
        double expected = log(
            einsum("i, ij, ijk, i, j, k->",
                _Q1.Cpt,
                _Q2.Cpt,
                _Y.Cpt,
                _evidence.GetState(_Q1).AsTensor(),
                _evidence.GetState(_Q2).AsTensor(),
                _evidence.GetState(_Y).AsTensor()))
            .item<double>();

        // Act
        double actual = _sut.LogLikelihood;

        // Assert
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void InferSingleNodeWithParents_AllObserved_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = einsum("i, ij, ijk, i, j, k->ij",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Q1).AsTensor(),
            _evidence.GetState(_Q2).AsTensor(),
            _evidence.GetState(_Y).AsTensor());
        pQ1xQ2_expected /= pQ1xQ2_expected.sum();
        Tensor pQ1xQ2xY_expected = einsum("i, ij, ijk, i, j, k->ijk",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Q1).AsTensor(),
            _evidence.GetState(_Q2).AsTensor(),
            _evidence.GetState(_Y).AsTensor());
        pQ1xQ2xY_expected /= pQ1xQ2xY_expected.sum();

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ1xQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            AssertHelpers.AssertTensorEqual(pQ1xQ2_actual, pQ1xQ2_expected);
            AssertHelpers.AssertTensorEqual(pQ1xQ2xY_actual, pQ1xQ2xY_expected);
        });
    }
}