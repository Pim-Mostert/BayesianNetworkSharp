using TorchSharp;

namespace BayesianNetwork.Inference.GenericTests.Helpers;

internal static class AssertHelpers
{
    public static void AssertTensorEqual(torch.Tensor actual, torch.Tensor expected, double tolerance = 1e-5)
    {
        var actualArray = actual.data<double>().ToArray();
        var expectedArray = expected.data<double>().ToArray();

        Assert.That(actualArray, Is.EqualTo(expectedArray).Within(tolerance));
    }
}
