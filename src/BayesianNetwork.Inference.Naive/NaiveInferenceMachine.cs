using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NaiveInferenceMachine : IInferenceMachine
{
    private readonly BayesianNetwork _bayesianNetwork;
    private readonly IDictionary<Node, int> _nodeIndex;
    private readonly Tensor _pComplete;

    public NaiveInferenceMachine(BayesianNetwork bayesianNetwork)
    {
        _bayesianNetwork = bayesianNetwork;
        _nodeIndex = _bayesianNetwork
            .Nodes
            .Select((n, i) => (Node: n, Index: i))
            .ToDictionary(
                ni => ni.Node,
                ni => ni.Index);

        _pComplete = CalculatePComplete();
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

    private Tensor Infer(IList<Node> nodes)
    {
        var nodeIndices = nodes.Select(n => _nodeIndex[n]).ToHashSet();
        long[] dimsToSumOver = Enumerable.Range(0, _bayesianNetwork.Nodes.Count)
            .Where(i => !nodeIndices.Contains(i))
            .Select(i => (long)i)
            .ToArray();

        return _pComplete.sum(dim: dimsToSumOver);
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
