using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.SumProduct.Test;

[TestFixture]
public class ComplexNetworkWithSingleParents_SomeObserved : GenericTests.ComplexNetworkWithSingleParents_SomeObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new SumProductInferenceMachine(bayesianNetwork);
    }
}