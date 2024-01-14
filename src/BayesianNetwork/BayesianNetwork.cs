namespace BayesianNetwork;

public class BayesianNetwork(IList<Node> nodes, IDictionary<Node, IList<Node>> parents)
{
    public IReadOnlyList<Node> Nodes { get; init; } = nodes.AsReadOnly();
    public IDictionary<Node, IList<Node>> Parents { get; init; } = parents;

    public int NumNodes => Nodes.Count;
}
