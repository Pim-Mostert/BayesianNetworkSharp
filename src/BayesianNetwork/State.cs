using TorchSharp;

namespace BayesianNetwork;

public class State(double[] tensor)
{
    private readonly torch.Tensor _tensor = torch.tensor(tensor);

    public long NumDims => _tensor.shape.Single();
    public torch.Tensor AsTensor() => _tensor;
}
