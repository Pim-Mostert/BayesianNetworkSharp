namespace BayesianNetwork;

public class Evidence(Dictionary<Node, State> states)
{
    private readonly Dictionary<Node, State> _states = states;

    public State GetState(Node node) => _states[node];
}
