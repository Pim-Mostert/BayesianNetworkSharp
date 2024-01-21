using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class HandleNumericalUnderflow
{
    private Node _Q;
    private Node[] _Ys;
    private NaiveInferenceMachine _sut;
    private Evidence _evidence;

    [SetUp]
    public void Setup()
    {
        torch.set_default_dtype(torch.float64);

        _Q = new Node(cpt: torch.tensor(new[] { 0.5, 0.5 }), name: "Q");
        _Ys = Enumerable.Range(0, 10)
            .Select(i =>
                new Node(cpt: torch.tensor(new[,] { { 1e-100, 1 - 1e-100 }, { 1 - 1e-100, 1e-100 } }), parents: [_Q], name: "Y", isObserved: true))
            .ToArray();

        BayesianNetwork bayesianNetwork = new(nodes: [_Q, .. _Ys]);

        _sut = new NaiveInferenceMachine(bayesianNetwork);

        EvidenceBuilder evidenceBuilder = EvidenceBuilder.For(bayesianNetwork);
        foreach (var y in _Ys)
            evidenceBuilder.SetState(y, new State([1 - 1E-100, 1E-100]));

        _evidence = evidenceBuilder.Build();

        _sut = new NaiveInferenceMachine(bayesianNetwork);
        _sut.EnterEvidence(_evidence);
    }

    [Test]
    public void InferSingleNode_SingleNodeObserved_CorrectInference()
    {
        // Assign

        // Act
        Tensor pQ_actual = _sut.Infer(_Q);
        IEnumerable<Tensor> pYs_actual = _Ys.Select(y => _sut.Infer(y));

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(pQ_actual.data<double>(), Is.All.Not.NaN);
            foreach (var y_actual in pYs_actual)
                Assert.That(y_actual.data<double>(), Is.All.Not.NaN);
        });
    }

    // [Test]
    // public void LogLikelihood_SingleNodeObserved_Correct()
    // {
    //     // Assign
    //     double expected = torch.log(
    //         torch.einsum("i, ij, jk, k->",
    //             _Q1.Cpt,
    //             _Q2.Cpt,
    //             _Y.Cpt,
    //             _evidence.GetState(_Y).AsTensor()))
    //         .item<double>();

    //     // Act
    //     double actual = _sut.LogLikelihood;

    //     // Assert
    //     Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    // }

    // [Test]
    // public void InferSingleNodeWithParents_SingleNodeObserved_CorrectInference()
    // {
    //     // Assign
    //     Tensor pQ1xQ2_expected = torch.einsum("i, ij, jk, k->ij",
    //         _Q1.Cpt,
    //         _Q2.Cpt,
    //         _Y.Cpt,
    //         _evidence.GetState(_Y).AsTensor());
    //     pQ1xQ2_expected /= pQ1xQ2_expected.sum();
    //     Tensor pQ2xY_expected = torch.einsum("i, ij, jk, k->jk",
    //         _Q1.Cpt,
    //         _Q2.Cpt,
    //         _Y.Cpt,
    //         _evidence.GetState(_Y).AsTensor());
    //     pQ2xY_expected /= pQ2xY_expected.sum();

    //     // Act
    //     Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
    //     Tensor pQ2xY_actual = _sut.Infer(_Y, includeParents: true);

    //     // Assert
    //     Assert.Multiple(() =>
    //     {
    //         Helpers.AssertTensorEqual(pQ1xQ2_actual, pQ1xQ2_expected);
    //         Helpers.AssertTensorEqual(pQ2xY_actual, pQ2xY_expected);
    //     });
    // }
}