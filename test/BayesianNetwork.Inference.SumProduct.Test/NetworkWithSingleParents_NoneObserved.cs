using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.SumProduct.Test;

[TestFixture]
public class NetworkWithSingleParents_NoneObserved : GenericTests.NetworkWithSingleParents_NoneObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new SumProductInferenceMachine(bayesianNetwork);
    }
}