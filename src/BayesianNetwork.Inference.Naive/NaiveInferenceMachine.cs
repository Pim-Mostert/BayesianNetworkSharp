using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NaiveInferenceMachine : IInferenceMachine
{
    private readonly BayesianNetwork _bayesianNetwork;

    private readonly IDictionary<Node, int> _nodeIndex;

    private readonly Tensor _pPrior;
    private Tensor _pPosterior;
    private Tensor _pEvidence;
    private double? _logLikelihood;

    public NaiveInferenceMachine(BayesianNetwork bayesianNetwork)
    {
        _bayesianNetwork = bayesianNetwork;
        _nodeIndex = _bayesianNetwork
            .Nodes
            .Select((n, i) => (Node: n, Index: i))
            .ToDictionary(
                ni => ni.Node,
                ni => ni.Index);

        _pPrior = CalculatePPrior();
        _pEvidence = torch.ones(_pPrior.shape, dtype: torch.float64);
        _pPosterior = _pPrior;
    }

    public Tensor Infer(Node node, bool includeParents = false)
    {
        if (includeParents)
        {
            var parents = node.Parents;
            return Infer([.. parents, node]);
        }
        else
        {
            return Infer([node]);
        }
    }

    public void EnterEvidence(Evidence evidence)
    {
        long[] allDimensions = _bayesianNetwork
            .Nodes
            .Select(n => n.NumStates)
            .ToArray();

        foreach (Node observedNode in _bayesianNetwork.ObservedNodes)
        {
            long[] newShape = Enumerable.Repeat<long>(1, _bayesianNetwork.NumNodes).ToArray();
            newShape[_nodeIndex[observedNode]] = observedNode.NumStates;

            _pEvidence *= evidence.GetState(observedNode).AsTensor().reshape(newShape);
        }

        _pPosterior = _pPrior * _pEvidence;

        var c = _pPosterior.sum();
        _pPosterior /= c;

        _logLikelihood = torch.log(c).item<double>();
    }

    public double LogLikelihood
    {
        get
        {
            if (!_logLikelihood.HasValue)
                throw new InvalidOperationException("Evidence must be entered first");

            return _logLikelihood.Value;
        }
    }

    private Tensor Infer(IList<Node> nodes)
    {
        var nodeIndices = nodes.Select(n => _nodeIndex[n]).ToHashSet();
        long[] dimsToSumOver = Enumerable.Range(0, _bayesianNetwork.Nodes.Count)
            .Where(i => !nodeIndices.Contains(i))
            .Select(i => (long)i)
            .ToArray();

        return dimsToSumOver.Length == 0
            ? _pPosterior
            : _pPosterior.sum(dim: dimsToSumOver);
    }

    private Tensor CalculatePPrior()
    {
        long[] allDimensions = _bayesianNetwork
            .Nodes
            .Select(n => n.NumStates)
            .ToArray();

        Tensor pPrior = torch.ones(allDimensions);

        foreach (Node node in _bayesianNetwork.Nodes)
        {
            long[] newShape = Enumerable.Repeat<long>(1, _bayesianNetwork.NumNodes).ToArray();
            newShape[_nodeIndex[node]] = node.NumStates;

            foreach (var parent in node.Parents)
            {
                newShape[_nodeIndex[parent]] = parent.NumStates;
            }

            pPrior *= node.Cpt.reshape(newShape);
        }

        pPrior /= pPrior.sum();

        return pPrior;
    }
}
