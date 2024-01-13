using TorchSharp;

namespace BayesianNetwork;

public class Node
{
    public required torch.Tensor Cpt { get; set; }
    public long NumStates => Cpt.shape.Last();

    public string Name { get; init; } = "<unnamed>";
    public override string ToString() => Name;
}