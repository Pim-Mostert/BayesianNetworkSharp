using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

[TestFixture]
public class NetworkWithMultipleParents_NoneObserved : GenericTests.NetworkWithMultipleParents_NoneObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}