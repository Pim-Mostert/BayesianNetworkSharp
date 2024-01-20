using TorchSharp;

namespace BayesianNetwork;

public class State(double[] vector)
{
    private readonly torch.Tensor _vector = torch.tensor(vector, dtype: torch.float64);

    public long NumDims => _vector.shape.Single();
    public torch.Tensor AsTensor() => _vector;
}
