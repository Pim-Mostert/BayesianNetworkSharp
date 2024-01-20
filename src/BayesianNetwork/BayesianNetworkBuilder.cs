namespace BayesianNetwork;

public class BayesianNetworkBuilder
{
    private readonly HashSet<Node> _nodes = [];
    private readonly HashSet<Node> _observedNodes = [];
    private readonly Dictionary<Node, ISet<Node>> _parents = [];

    public BayesianNetworkBuilder AddObservedNode(Node node, Node? parent = null)
    {
        AddObservedNode(
            node,
            parent is null ?
                new HashSet<Node>()
                : [parent]);

        return this;
    }

    public BayesianNetworkBuilder AddObservedNode(Node node, ISet<Node> parents)
    {
        if (!_observedNodes.Add(node))
            throw new InvalidOperationException($"Node {node} was already added as observed node");

        AddNode(node, parents);

        return this;
    }

    public BayesianNetworkBuilder AddNode(Node node, Node? parent = null)
    {
        AddNode(
            node,
            parent is null ?
                new HashSet<Node>()
                : [parent]);

        return this;
    }

    public BayesianNetworkBuilder AddNode(Node node, ISet<Node> parents, bool isObserved = false)
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

        return new BayesianNetwork(_nodes, _parents, _observedNodes);
    }
}