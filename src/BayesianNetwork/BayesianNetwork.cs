using System.Collections.Frozen;

namespace BayesianNetwork;

public class BayesianNetwork(
    ISet<Node> nodes,
    IDictionary<Node, ISet<Node>> parents,
    ISet<Node>? observedNodes = null)
{
    public IReadOnlySet<Node> Nodes { get; init; } = nodes.ToFrozenSet();
    public IReadOnlySet<Node> ObservedNodes { get; init; } =
        (observedNodes ?? new HashSet<Node>()).ToFrozenSet();
    public IReadOnlyDictionary<Node, IReadOnlySet<Node>> Parents { get; init; } = parents
        .ToFrozenDictionary(
            kvp => kvp.Key,
            kvp => (IReadOnlySet<Node>)kvp.Value.ToFrozenSet());

    public int NumNodes => Nodes.Count;
}
