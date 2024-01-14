using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Abstractions;

public interface IInferenceMachine
{
    Tensor Infer(Node node, bool includeParents = false);
}