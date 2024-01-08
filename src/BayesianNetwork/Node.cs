using TorchSharp;

namespace BayesianNetwork;

public class Node
{
    public required torch.Tensor Cpt { get; set; }
}