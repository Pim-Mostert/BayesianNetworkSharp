using TorchSharp;

namespace BayesianNetwork;

public class Node
{
    public Node(torch.Tensor cpt, List<Node>? parents = null, bool isObserved = false, string name = "<unnamed>")
    {
        parents ??= [];

        if (parents.Count != parents.Distinct().Count())
            throw new ArgumentException("May only contain unique elements", nameof(parents));

        if (parents.Count != (cpt.shape.Length - 1))
            throw new ArgumentException("Size of cpt does not match number of parents", nameof(cpt));

        if (parents.Zip(cpt.shape[..^1]).Any(x => x.First.NumStates != x.Second))
            throw new ArgumentException("Dimensionality of cpt does not match number of states of parents", nameof(cpt));

        Parents = parents;
        Cpt = cpt;
        IsObserved = isObserved;
        Name = name;
    }

    public IReadOnlyList<Node> Parents { get; init; }
    public torch.Tensor Cpt { get; init; }
    public bool IsObserved { get; init; }

    public long NumStates => Cpt.shape.Last();

    public string Name { get; init; }
    public override string ToString() => Name;
}