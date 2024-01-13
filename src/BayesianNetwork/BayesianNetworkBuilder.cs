namespace BayesianNetwork;

public class BayesianNetworkBuilder
{
    private ISet<Node> _nodes { get; init; } = new HashSet<Node>();
    private Dictionary<Node, IEnumerable<Node>> _parents { get; init; } = [];

    private bool _hasRootNode = false;

    public BayesianNetworkBuilder AddRootNode(Node node)
    {
        if (_hasRootNode)
            throw new InvalidOperationException($"A root node was already added");

        if (!_nodes.Add(node))
            throw new InvalidOperationException($"Node {node} was already added");

        _hasRootNode = true;

        _parents[node] = [];

        return this;
    }

    public BayesianNetworkBuilder AddNode(Node node, Node parent)
    {
        if (!_nodes.Contains(parent))
            throw new InvalidOperationException($"The parent node {parent} is not part of the bayesian network");

        if (!_nodes.Add(node))
            throw new InvalidOperationException($"Node {node} was already added");

        _parents[node] = [parent];

        return this;
    }

    public BayesianNetwork Build()
    {
        return new BayesianNetwork(_nodes, _parents);
    }
}