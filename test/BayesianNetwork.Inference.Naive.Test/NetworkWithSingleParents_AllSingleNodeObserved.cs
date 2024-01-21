using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NetworkWithSingleParents_SingleNodeObserved
{
    private Node _Q1;
    private Node _Q2;
    private Node _Y;

    private BayesianNetwork _bayesianNetwork;

    private NaiveInferenceMachine _sut;
    private Evidence _evidence;

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
            .AddObservedNode(_Y, parent: _Q2)
            .Build();

        _evidence = EvidenceBuilder.For(_bayesianNetwork)
            .SetState(_Y, new State([1, 0]))
            .Build();

        _sut = new NaiveInferenceMachine(_bayesianNetwork);
        _sut.EnterEvidence(_evidence);
    }

    [Test]
    public void InferSingleNode_SingleNodeObserved_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = torch.einsum("i, ij, jk, k->i",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Y).AsTensor());
        pQ1_expected /= pQ1_expected.sum();
        Tensor pQ2_expected = torch.einsum("i, ij, jk, k->j",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Y).AsTensor());
        pQ2_expected /= pQ2_expected.sum();
        Tensor pY_expected = torch.einsum("i, ij, jk, k->k",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Y).AsTensor());
        pY_expected /= pY_expected.sum();

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pY_actual = _sut.Infer(_Y);

        // Assert
        Assert.Multiple(() =>
        {
            Helpers.AssertTensorEqual(pQ1_actual, pQ1_expected);
            Helpers.AssertTensorEqual(pQ2_actual, pQ2_expected);
            Helpers.AssertTensorEqual(pY_actual, pY_expected);
        });
    }

    [Test]
    public void LogLikelihoodSingleNodeObserved_Correct()
    {
        // Assign
        double expected = torch.log(
            torch.einsum("i, ij, jk, k->",
                _Q1.Cpt,
                _Q2.Cpt,
                _Y.Cpt,
                _evidence.GetState(_Y).AsTensor()))
            .item<double>();

        // Act
        double actual = _sut.LogLikelihood;

        // Assert
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void InferSingleNodeWithParents_SingleNodeObserved_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = torch.einsum("i, ij, jk, k->ij",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Y).AsTensor());
        pQ1xQ2_expected /= pQ1xQ2_expected.sum();
        Tensor pQ2xY_expected = torch.einsum("i, ij, jk, k->jk",
            _Q1.Cpt,
            _Q2.Cpt,
            _Y.Cpt,
            _evidence.GetState(_Y).AsTensor());
        pQ2xY_expected /= pQ2xY_expected.sum();

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            Helpers.AssertTensorEqual(pQ1xQ2_actual, pQ1xQ2_expected);
            Helpers.AssertTensorEqual(pQ2xY_actual, pQ2xY_expected);
        });
    }
}