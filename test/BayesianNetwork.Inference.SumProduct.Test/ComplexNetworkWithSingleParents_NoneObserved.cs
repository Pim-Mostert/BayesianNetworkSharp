using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.SumProduct.Test;

[TestFixture]
public class ComplexNetworkWithSingleParents_NoneObserved : GenericTests.ComplexNetworkWithSingleParents_NoneObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new SumProductInferenceMachine(bayesianNetwork);
    }
}