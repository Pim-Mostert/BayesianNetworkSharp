using BayesianNetwork.Inference.Abstractions;
using TorchSharp;
using static TorchSharp.torch;

namespace BayesianNetwork.Inference.Naive;

public class NetworkWithSingleParents
{
    private Node _Q1;
    private Node _Q2;
    private Node _Y;

    private BayesianNetwork _bayesianNetwork;

    private IInferenceMachine _sut;

    [SetUp]
    public void Setup()
    {
        _Q1 = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2]),
            Name = "Q1"
        };
        _Q2 = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2, 2]),
            Name = "Q2"
        };
        _Y = new Node
        {
            Cpt = GenerateRandomProbabilityMatrix([2, 2]),
            Name = "Y"
        };

        _bayesianNetwork = new BayesianNetworkBuilder()
            .AddRootNode(_Q1)
            .AddNode(_Q2, parent: _Q1)
            .AddNode(_Y, parent: _Q2)
            .Build();

        _sut = new NaiveInferenceMachine(_bayesianNetwork);
    }

    [Test]
    public void InferSingleNode_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1_expected = torch.einsum("i->i", _Q1.Cpt);
        Tensor pQ2_expected = torch.einsum("i, ij->j", _Q1.Cpt, _Q2.Cpt);
        Tensor pY_expected = torch.einsum("i, ij, jk->k", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1_actual = _sut.Infer(_Q1);
        Tensor pQ2_actual = _sut.Infer(_Q2);
        Tensor pY_actual = _sut.Infer(_Y);

        // Assert
        Assert.Multiple(() =>
        {
            AssertTensorEqual(pQ1_expected, pQ1_actual);
            AssertTensorEqual(pQ2_expected, pQ2_actual);
            AssertTensorEqual(pY_expected, pY_actual);
        });
    }

    [Test]
    public void InferSingleNodeWithParents_NoObservations_CorrectInference()
    {
        // Assign
        Tensor pQ1xQ2_expected = torch.einsum("i, ij->ij", _Q1.Cpt, _Q2.Cpt);
        Tensor pQ2xY_expected = torch.einsum("i, ij, jk->jk", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act
        Tensor pQ1xQ2_actual = _sut.Infer(_Q2, includeParents: true);
        Tensor pQ2xY_actual = _sut.Infer(_Y, includeParents: true);

        // Assert
        Assert.Multiple(() =>
        {
            AssertTensorEqual(pQ1xQ2_expected, pQ1xQ2_actual);
            AssertTensorEqual(pQ2xY_expected, pQ2xY_actual);
        });
    }

    // def test_all_observed_single_nodes(self):
    //     # Assign
    //     device = self.get_torch_settings().device
    //     dtype = self.get_torch_settings().dtype

    //     evidence = rescale_tensors([
    //         torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
    //         torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
    //         torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
    //     ])
    //     num_observations = evidence[0].shape[0]

    //     p_Q1_expected = torch.einsum('i, ij, jk, ni, nj, nk->ni', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence)
    //     p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
    //     p_Q2_expected = torch.einsum('i, ij, jk, ni, nj, nk->nj', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence)
    //     p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
    //     p_Y_expected = torch.einsum('i, ij, jk, ni, nj, nk->nk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence)
    //     p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

    //     # Act
    //     sut = self.create_inference_machine(
    //         bayesian_network=self.network,
    //         observed_nodes=[self.Q1, self.Q2, self.Y],
    //         num_observations=num_observations)

    //     sut.enter_evidence(evidence)

    //     [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

    //     # Assert
    //     self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
    //     self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
    //     self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

    private static void AssertTensorEqual(Tensor expected, Tensor actual, double tolerance = 1e-5)
    {
        var expectedArray = expected.data<float>().ToArray();
        var actualArray = actual.data<float>().ToArray();

        Assert.That(expectedArray, Is.EqualTo(actualArray).Within(tolerance));
    }

    private static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size);

        return p / p.sum(dim: -1, keepdim: true);
    }
}