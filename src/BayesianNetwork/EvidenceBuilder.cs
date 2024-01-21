namespace BayesianNetwork;

public class EvidenceBuilder
{
    private readonly IReadOnlyList<Node> _observedNodes;
    private readonly Dictionary<Node, State> _states = [];

    public static EvidenceBuilder For(BayesianNetwork bayesianNetwork) => new(bayesianNetwork);

    private EvidenceBuilder(BayesianNetwork bayesianNetwork)
    {
        _observedNodes = bayesianNetwork.ObservedNodes;
    }

    public EvidenceBuilder SetState(Node node, State state)
    {
        if (!_observedNodes.Contains(node))
            throw new InvalidOperationException($"Node {node} is not an observed node");

        if (state.NumDims != node.NumStates)
            throw new InvalidOperationException($"State cannot be assigned to node - dimensionality mismatch");

        if (_states.ContainsKey(node))
            throw new InvalidOperationException($"State for node {node} has already been set");

        _states[node] = state;

        return this;
    }

    public Evidence Build()
    {
        if (_observedNodes.Any(n => !_states.ContainsKey(n)))
            throw new InvalidOperationException("Not all observed nodes have a state set");

        return new Evidence(_states);
    }
}
