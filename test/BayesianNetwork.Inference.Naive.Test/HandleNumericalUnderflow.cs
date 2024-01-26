using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive.Test;

[TestFixture]
public class HandleNumericalUnderflow : GenericTests.HandleNumericalUnderflow
{
    protected override IInferenceMachine InferenceMachineFactory(BayesianNetwork bayesianNetwork)
    {
        return new NaiveInferenceMachine(bayesianNetwork);
    }
}