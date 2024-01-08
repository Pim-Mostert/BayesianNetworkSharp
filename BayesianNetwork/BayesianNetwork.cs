namespace BayesianNetwork;

public class BayesianNetwork
{
    public IReadOnlySet<Node> Nodes { get; init; }
    public IDictionary<Node, Node> Parents { get; init; }
    public IReadOnlyDictionary<Node, Node[]> Children { get; init; }

    public BayesianNetwork(ISet<Node> nodes, IDictionary<Node, Node> parents)
    {
        Nodes = new HashSet<Node>(nodes);
        Parents = parents;
        Children = nodes.ToDictionary(
            n => n,
            n => parents
                    .Where(kvp => kvp.Value == n)
                    .Select(kvp => kvp.Key)
                    .ToArray());
    }
}
