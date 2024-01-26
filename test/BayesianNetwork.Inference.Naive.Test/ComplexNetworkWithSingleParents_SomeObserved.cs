using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

[TestFixture]
public class ComplexNetworkWithSingleParents_SomeObserved : GenericTests.ComplexNetworkWithSingleParents_SomeObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}