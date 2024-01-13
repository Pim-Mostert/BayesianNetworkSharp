namespace BayesianNetwork;

public class BayesianNetwork(ISet<Node> nodes, IDictionary<Node, IEnumerable<Node>> parents)
{
    public IReadOnlySet<Node> Nodes { get; init; } = new HashSet<Node>(nodes);
    public IDictionary<Node, IEnumerable<Node>> Parents { get; init; } = parents;
}
