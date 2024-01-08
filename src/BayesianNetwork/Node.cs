using TorchSharp;

namespace BayesianNetwork;

public class Node
{
    public string Name { get; init; } = "<unnamed>";
    public required torch.Tensor Cpt { get; set; }

    public override string ToString() => Name;
}