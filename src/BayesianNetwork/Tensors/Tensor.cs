using TorchSharp;

namespace BayesianNetwork.Tensors;

public abstract class Tensor
{
    public torch.Tensor Value { get; private set; }

    public Tensor(torch.Tensor value, long numDims)
    {
        if (value.shape.Length != numDims)
            throw new ArgumentException($"Tensor was expected to have {numDims} dimensions", nameof(value));

        Value = value;
    }

    public static implicit operator torch.Tensor(Tensor tensor) => tensor.Value;
}

public class Tensor<D1>(torch.Tensor value) : Tensor(value, numDims: 1)
{
    public static implicit operator Tensor<D1>(torch.Tensor value) => new(value);
    public static implicit operator torch.Tensor(Tensor<D1> tensor) => tensor.Value;
}

// public class Tensor<D1, D2>(torch.Tensor value) : Tensor(value, numDims: 2) { }
// public class Tensor<D1, D2, D3>(torch.Tensor value) : Tensor(value, numDims: 3) { }
// public class Tensor<D1, D2, D3, D4>(torch.Tensor value) : Tensor(value, numDims: 4) { }
// public class Tensor<D1, D2, D3, D4, D5>(torch.Tensor value) : Tensor(value, numDims: 5) { }
// public class Tensor<D1, D2, D3, D4, D5, D6>(torch.Tensor value) : Tensor(value, numDims: 6) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7>(torch.Tensor value) : Tensor(value, numDims: 7) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8>(torch.Tensor value) : Tensor(value, numDims: 8) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9>(torch.Tensor value) : Tensor(value, numDims: 9) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10>(torch.Tensor value) : Tensor(value, numDims: 10) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11>(torch.Tensor value) : Tensor(value, numDims: 11) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12>(torch.Tensor value) : Tensor(value, numDims: 12) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13>(torch.Tensor value) : Tensor(value, numDims: 13) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14>(torch.Tensor value) : Tensor(value, numDims: 14) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15>(torch.Tensor value) : Tensor(value, numDims: 15) { }
// public class Tensor<D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16>(torch.Tensor value) : Tensor(value, numDims: 16) { }
