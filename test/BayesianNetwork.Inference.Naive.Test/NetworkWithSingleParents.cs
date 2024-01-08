using BayesianNetwork.Inference.Abstractions;
using TorchSharp;

namespace BayesianNetwork.Inference.Naive;

public class NetworkWithSingleParents
{
    private Node _Q1 { get; set; }
    private Node _Q2 { get; set; }
    private Node _Y { get; set; }

    private BayesianNetwork _bayesianNetwork { get; set; }

    private IInferenceMachine _sut { get; set; }

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
        var p_Q1_expected = torch.einsum("i->i", _Q1.Cpt);
        var p_Q2_expected = torch.einsum("i, ij->j", _Q1.Cpt, _Q2.Cpt);
        var p_Y_expected = torch.einsum("i, ij, jk->k", _Q1.Cpt, _Q2.Cpt, _Y.Cpt);

        // Act

        // # Assign
        // p_Q1_expected = torch.einsum('i->i', self.Q1.cpt)[None, ...]
        // p_Q2_expected = torch.einsum('i, ij->j', self.Q1.cpt, self.Q2.cpt)[None, ...]
        // p_Y_expected = torch.einsum('i, ij, jk->k', self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[None, ...]

        // # Act
        // sut = self.create_inference_machine(
        //     bayesian_network=self.network,
        //     observed_nodes=[self.Q2],
        //     num_observations=0)

        // [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

        // # Assert
        // self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
        // self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
        // self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)
    }

    private static torch.Tensor GenerateRandomProbabilityMatrix(long[] size)
    {
        var p = torch.rand(size);

        return p / p.sum(dim: -1, keepdim: true);
    }
}