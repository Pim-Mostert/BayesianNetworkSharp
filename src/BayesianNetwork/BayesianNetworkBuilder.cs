namespace BayesianNetwork;

public class BayesianNetworkBuilder
{
    private readonly HashSet<Node> _nodes = [];
    private readonly Dictionary<Node, ISet<Node>> _parents = [];

    public BayesianNetworkBuilder AddNode(Node node, Node? parent = null)
    {
        return AddNode(
            node,
            parent is null ?
                new HashSet<Node>()
                : [parent]);
    }

    public BayesianNetworkBuilder AddNode(Node node, ISet<Node> parents)
    {
        if (!_nodes.Add(node))
            throw new InvalidOperationException($"Node {node} was already added");

        if (!parents.IsSubsetOf(_nodes))
            throw new InvalidOperationException("Parents should be part of the network");

        _parents[node] = parents;

        return this;
    }

    public BayesianNetwork Build()
    {
        if (!_parents.Values.All(x => x.IsSubsetOf(_nodes)))
            throw new InvalidOperationException("All parent nodes should be part of the network");

        return new BayesianNetwork(_nodes, _parents);
    }
}