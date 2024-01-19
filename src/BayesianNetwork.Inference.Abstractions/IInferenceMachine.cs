using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Abstractions;

public interface IInferenceMachine
{
    void EnterEvidence(Evidence evidence);
    public double LogLikelihood { get; }
    Tensor Infer(Node node, bool includeParents = false);
}