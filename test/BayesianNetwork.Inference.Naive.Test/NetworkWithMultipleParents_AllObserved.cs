using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

[TestFixture]
public class NetworkWithMultipleParents_AllObserved : GenericTests.NetworkWithMultipleParents_AllObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}