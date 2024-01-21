namespace BayesianNetwork;

public class BayesianNetwork
{
    public BayesianNetwork(IList<Node> nodes)
    {
        if (nodes.Count != nodes.Distinct().Count())
            throw new ArgumentException("Nodes must be unique", nameof(nodes));

        var allParents = nodes.SelectMany(n => n.Parents).Distinct();
        if (allParents.Any(p => !nodes.Contains(p)))
            throw new ArgumentException("Not all of the nodes' parents are nodes themselves", nameof(nodes));

        Nodes = nodes.AsReadOnly();
        ObservedNodes = Nodes.Where(n => n.IsObserved).ToList().AsReadOnly();
    }

    public IReadOnlyList<Node> Nodes { get; init; }
    public IReadOnlyList<Node> ObservedNodes { get; init; }

    public int NumNodes => Nodes.Count;

}
