using TorchSharp;

namespace BayesianNetwork.Inference.GenericTests.Helpers;

internal static class TensorHelpers
{
    public static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size);

        return p / p.sum(dim: -1, keepdim: true);
    }
}