using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

public class ComplexNetworkWithSingleParents_NoneObserved : GenericTests.ComplexNetworkWithSingleParents_NoneObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}