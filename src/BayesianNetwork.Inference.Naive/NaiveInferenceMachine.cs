using BayesianNetwork.Inference.Abstractions;

namespace BayesianNetwork.Inference.Naive;

public class NaiveInferenceMachine(BayesianNetwork BayesianNetwork) : IInferenceMachine;
