namespace BayesianNetwork;

public class BayesianNetwork(ISet<Node> nodes, IDictionary<Node, Node> parents)
{
    public IReadOnlySet<Node> Nodes { get; init; } = new HashSet<Node>(nodes);
    public IDictionary<Node, Node> Parents { get; init; } = parents;
    public IReadOnlyDictionary<Node, Node[]> Children { get; init; } =
        nodes.ToDictionary(
            n => n,
            n => parents
                    .Where(kvp => kvp.Value == n)
                    .Select(kvp => kvp.Key)
                    .ToArray());
}
