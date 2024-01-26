using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

[TestFixture]
public class NetworkWithMultipleParents_SingleNodeObserved : GenericTests.NetworkWithMultipleParents_SingleNodeObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}