﻿using System.Collections.Generic;
using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NaiveInferenceMachine : IInferenceMachine
{
    private readonly BayesianNetwork _bayesianNetwork;
    private readonly ISet<Node> _observedNodes;

    private readonly IDictionary<Node, int> _nodeIndex;

    private readonly Tensor _pPrior;
    private Tensor _pPosterior;
    private Tensor _pEvidence;
    private double? _logLikelihood;

    public NaiveInferenceMachine(
        BayesianNetwork bayesianNetwork,
        ISet<Node>? observedNodes = null)
    {
        _bayesianNetwork = bayesianNetwork;
        _observedNodes = observedNodes ?? new HashSet<Node>();
        _nodeIndex = _bayesianNetwork
            .Nodes
            .Select((n, i) => (Node: n, Index: i))
            .ToDictionary(
                ni => ni.Node,
                ni => ni.Index);

        _pPrior = CalculatePComplete();
        _pEvidence = torch.ones(_pPrior.shape);
        _pPosterior = _pPrior;
    }

    public Tensor Infer(Node node, bool includeParents = false)
    {
        if (includeParents)
        {
            var parents = _bayesianNetwork.Parents[node];
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

        foreach (Node observedNode in _observedNodes)
        {
            long[] newShape = Enumerable.Repeat<long>(1, _bayesianNetwork.NumNodes).ToArray();
            newShape[_nodeIndex[observedNode]] = observedNode.NumStates;

            _pEvidence *= evidence[observedNode].reshape(newShape);
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

        return _pPosterior.sum(dim: dimsToSumOver);
    }

    private Tensor CalculatePComplete()
    {
        long[] allDimensions = _bayesianNetwork
            .Nodes
            .Select(n => n.NumStates)
            .ToArray();

        Tensor pComplete = torch.ones(allDimensions);

        foreach (Node node in _bayesianNetwork.Nodes)
        {
            long[] newShape = Enumerable.Repeat<long>(1, _bayesianNetwork.NumNodes).ToArray();
            newShape[_nodeIndex[node]] = node.NumStates;

            foreach (var parent in _bayesianNetwork.Parents[node])
            {
                newShape[_nodeIndex[parent]] = parent.NumStates;
            }

            pComplete *= node.Cpt.reshape(newShape);
        }

        return pComplete;
    }
}
