using TorchSharp;

namespace BayesianNetwork;

public class Evidence : Dictionary<Node, torch.Tensor>
{
}
