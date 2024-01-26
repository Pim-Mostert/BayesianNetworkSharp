using BayesianNetwork.Inference.Abstractions;
using TorchSharp;

namespace BayesianNetwork.Inference.GenericTests;

public abstract class HandleNumericalUnderflow
{
    private Node _Q;
    private Node[] _Ys;
    private IInferenceMachine _sut;
    private Evidence _evidence;

    protected abstract IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork);

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

        EvidenceBuilder evidenceBuilder = EvidenceBuilder.For(bayesianNetwork);
        foreach (var y in _Ys)
            evidenceBuilder.SetState(y, new State([1 - 1E-100, 1E-100]));

        _evidence = evidenceBuilder.Build();

        _sut = InferenceMachineFactory(bayesianNetwork);
        _sut.EnterEvidence(_evidence);
    }

    [Test]
    public void InferAllNodes_WithEvidence_NoNaNs()
    {
        // Assign

        // Act
        torch.Tensor pQ_actual = _sut.Infer(_Q);
        IEnumerable<torch.Tensor> pYs_actual = _Ys.Select(y => _sut.Infer(y));

        // Assert
        Assert.Multiple(() =>
        {
            Assert.That(pQ_actual.data<double>(), Is.All.Not.NaN);
            foreach (var y_actual in pYs_actual)
                Assert.That(y_actual.data<double>(), Is.All.Not.NaN);
        });
    }

    [Test]
    public void LogLikelihood_WithEvidence_NoNaNs()
    {
        // Assign

        // Act
        double actual = _sut.LogLikelihood;

        // Assert
        Assert.That(actual, Is.Not.NaN);
    }

    [Test]
    public void InferSingleNodeWithParents_SingleNodeObserved_CorrectInference()
    {
        // Assign

        // Act
        IEnumerable<torch.Tensor> pQ1xYs_actual = _Ys.Select(y => _sut.Infer(y, includeParents: true));

        // Assert
        Assert.Multiple(() =>
        {
            foreach (var pQ1xY_actual in pQ1xYs_actual)
                Assert.That(pQ1xY_actual.data<double>(), Is.All.Not.NaN);
        });
    }
}