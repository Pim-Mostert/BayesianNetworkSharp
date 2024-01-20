using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

internal static class Helpers
{
    public static void AssertTensorEqual(Tensor actual, Tensor expected, double tolerance = 1e-5)
    {
        var actualArray = actual.data<double>().ToArray();
        var expectedArray = expected.data<double>().ToArray();

        Assert.That(actualArray, Is.EqualTo(expectedArray).Within(tolerance));
    }

    public static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size, dtype: torch.float64);

        return p / p.sum(dim: -1, keepdim: true);
    }
}