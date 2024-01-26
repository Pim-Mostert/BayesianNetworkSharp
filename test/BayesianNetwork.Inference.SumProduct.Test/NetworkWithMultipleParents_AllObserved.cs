using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.SumProduct.Test;

[TestFixture]
public class NetworkWithMultipleParents_AllObserved : GenericTests.NetworkWithMultipleParents_AllObserved
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new SumProductInferenceMachine(bayesianNetwork);
    }
}