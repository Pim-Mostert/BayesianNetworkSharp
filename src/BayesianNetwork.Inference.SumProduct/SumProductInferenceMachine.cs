using BayesianNetwork.Inference.Abstractions;
using TorchSharp;

namespace BayesianNetwork.Inference.SumProduct;

public class SumProductInferenceMachine : IInferenceMachine
{
    public SumProductInferenceMachine(BayesianNetwork bayesianNetwork)
    {
    }

    public double LogLikelihood => throw new NotImplementedException();

    public void EnterEvidence(Evidence evidence)
    {
        throw new NotImplementedException();
    }

    public torch.Tensor Infer(Node node, bool includeParents = false)
    {
        throw new NotImplementedException();
    }
}
