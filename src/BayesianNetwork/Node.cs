using BayesianNetwork.Tensors;

namespace BayesianNetwork;

public class Node<T> where T : Tensor
{
    public Node(Tensor<State> cpt, List<Node>? parents = null, bool isObserved = false, string name = "<unnamed>")
    {
        parents ??= [];

        if (parents.Count != parents.Distinct().Count())
            throw new ArgumentException("May only contain unique elements", nameof(parents));

        if (parents.Count != (cpt.Value.shape.Length - 1))
            throw new ArgumentException("Size of cpt does not match number of parents", nameof(cpt));

        if (parents.Zip(cpt.Value.shape[..^1]).Any(x => x.First.NumStates != x.Second))
            throw new ArgumentException("Dimensionality of cpt does not match number of states of parents", nameof(cpt));

        Parents = parents;
        Cpt = cpt;
        IsObserved = isObserved;
        Name = name;
    }

    public IReadOnlyList<Node> Parents { get; init; }
    public Tensor<State> Cpt { get; init; }
    public bool IsObserved { get; init; }

    public long NumStates => Cpt.Value.shape.Last();

    public string Name { get; init; }
    public override string ToString() => Name;
}