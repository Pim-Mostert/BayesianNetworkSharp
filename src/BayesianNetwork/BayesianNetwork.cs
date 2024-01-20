using System.Collections.Frozen;

namespace BayesianNetwork;

public class BayesianNetwork(ISet<Node> nodes, IDictionary<Node, ISet<Node>> parents)
{
    public IReadOnlySet<Node> Nodes { get; init; } = nodes.ToFrozenSet();
    public IReadOnlyDictionary<Node, IReadOnlySet<Node>> Parents { get; init; } = parents
        .ToFrozenDictionary(
            kvp => kvp.Key,
            kvp => (IReadOnlySet<Node>)kvp.Value.ToFrozenSet());

    public int NumNodes => Nodes.Count;
}
