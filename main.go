package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// Hyperparameters
const (
	InnerLearningRate = 0.01
	OuterLearningRate = 0.001
	NumInnerSteps     = 5
	NumMetaIterations = 100
	NumTasks          = 10
	MaxReward         = 100
	MinReward         = -100
)

// Numerical stability constant for softmax
const Epsilon = 1e-10

// NetworkEnv represents the environment for network tasks
type NetworkEnv struct {
	data        [][]float64
	currentStep int
}

// NewNetworkEnv initializes an environment from CSV data
func NewNetworkEnv(data [][]float64) *NetworkEnv {
	return &NetworkEnv{data: data}
}

// Step performs a step in the environment based on the action taken
func (env *NetworkEnv) Step(action int) ([]float64, float64, bool) {
	row := env.data[env.currentStep]
	path1 := row[:3]
	path2 := row[3:6]

	var reward float64
	if action == 0 {
		reward = calculateReward(path1, path2, 0)
	} else {
		reward = calculateReward(path1, path2, 1)
	}

	env.currentStep++
	done := env.currentStep >= len(env.data)

	state := path1
	if action == 1 {
		state = path2
	}

	return state, reward, done
}

// Reset resets the environment to the initial state
func (env *NetworkEnv) Reset() []float64 {
	env.currentStep = 0
	row := env.data[env.currentStep]
	return row[:3]
}

// LoadAndCleanDataset loads and cleans the CSV dataset
func LoadAndCleanDataset(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data [][]float64
	columns := len(records[0])
	columnSums := make([]float64, columns)
	columnCounts := make([]float64, columns)

	// Parse and clean data
	for _, record := range records[1:] {
		row := make([]float64, columns)
		for i, val := range record {
			if v, err := strconv.ParseFloat(val, 64); err == nil {
				row[i] = v
				columnSums[i] += v
				columnCounts[i]++
			} else {
				row[i] = math.NaN() // Mark missing values as NaN
			}
		}
		data = append(data, row)
	}

	// Fill missing values with column means
	for i := range data {
		for j, val := range data[i] {
			if math.IsNaN(val) {
				data[i][j] = columnSums[j] / columnCounts[j] // Replace NaN with column mean
			}
		}
	}

	return data, nil
}

// Reward calculation function using weights
func calculateReward(path1, path2 []float64, action int) float64 {
	weights := []float64{0.5, 0.3, 0.2}

	// Calculate scores for both paths
	score1 := 0.0
	score2 := 0.0
	for i := range path1 {
		score1 += weights[i] * path1[i]
		score2 += weights[i] * path2[i]
	}

	// Reward logic
	var reward float64
	if action == 0 {
		reward = score2 - score1
	} else {
		reward = score1 - score2
	}

	// Clamp reward to [-100, 100]
	return math.Max(MinReward, math.Min(reward, MaxReward))
}

// Normalize rewards dynamically for better learning signals (Z-score normalization)
func normalizeRewards(rewards []float64) []float64 {
	mean := 0.0
	for _, r := range rewards {
		mean += r
	}
	mean /= float64(len(rewards))

	std := 0.0
	for _, r := range rewards {
		std += (r - mean) * (r - mean)
	}
	std = math.Sqrt(std / float64(len(rewards)))

	normalized := make([]float64, len(rewards))
	for i, r := range rewards {
		normalized[i] = (r - mean) / (std + Epsilon) // Avoid division by zero
	}
	return normalized
}

// PolicyNetwork defines a simple neural network with two layers and softmax output
type PolicyNetwork struct {
	fc1Weights *mat.Dense
	fc1Bias    *mat.Dense
	fc2Weights *mat.Dense
	fc2Bias    *mat.Dense
}

// NewPolicyNetwork initializes the PolicyNetwork with random weights and biases
func NewPolicyNetwork(inputSize, hiddenSize, outputSize int) *PolicyNetwork {
	return &PolicyNetwork{
		fc1Weights: randomMatrix(inputSize, hiddenSize),
		fc1Bias:    randomMatrix(1, hiddenSize),
		fc2Weights: randomMatrix(hiddenSize, outputSize),
		fc2Bias:    randomMatrix(1, outputSize),
	}
}

// Helper function to initialize a matrix with random values
func randomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
	}
	return mat.NewDense(rows, cols, data)
}

// Adds bias to each row in the output
func addBias(output, bias *mat.Dense) {
	output.Apply(func(i, j int, v float64) float64 {
		return v + bias.At(0, j)
	}, output)
}

// ReLU activation function
func applyReLU(m *mat.Dense) {
	m.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}, m)
}

// Softmax with numerical stability
func applySoftmax(m *mat.Dense) {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		row := mat.Row(nil, i, m)
		maxVal := max(row)
		sumExp := 0.0
		for j := 0; j < cols; j++ {
			row[j] = math.Exp(row[j] - maxVal)
			sumExp += row[j]
		}
		for j := 0; j < cols; j++ {
			m.Set(i, j, row[j]/(sumExp+Epsilon))
		}
	}
}

// Maximum value in a slice
func max(slice []float64) float64 {
	maxVal := slice[0]
	for _, val := range slice {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

// Forward performs the forward pass through the network
func (pn *PolicyNetwork) Forward(input *mat.Dense) *mat.Dense {
	var fc1Output mat.Dense
	fc1Output.Mul(input, pn.fc1Weights) // Compute W1 * input
	addBias(&fc1Output, pn.fc1Bias)     // Add bias b1
	applyReLU(&fc1Output)               // Apply ReLU activation

	var fc2Output mat.Dense
	fc2Output.Mul(&fc1Output, pn.fc2Weights) // Compute W2 * hidden_layer
	addBias(&fc2Output, pn.fc2Bias)          // Add bias b2
	applySoftmax(&fc2Output)                 // Apply Softmax activation

	return &fc2Output
}

// ClonePolicy creates a copy of the current policy network
func (pn *PolicyNetwork) Clone() *PolicyNetwork {
	return &PolicyNetwork{
		fc1Weights: mat.DenseCopyOf(pn.fc1Weights),
		fc1Bias:    mat.DenseCopyOf(pn.fc1Bias),
		fc2Weights: mat.DenseCopyOf(pn.fc2Weights),
		fc2Bias:    mat.DenseCopyOf(pn.fc2Bias),
	}
}

// SampleTrajectory collects a trajectory of states, actions, and rewards for a given environment
func SampleTrajectory(env *NetworkEnv, policy *PolicyNetwork, epsilon float64) ([]*mat.Dense, []int, []float64) {
	var states []*mat.Dense
	var actions []int
	var rewards []float64

	state := env.Reset()
	done := false

	for !done {
		stateTensor := mat.NewDense(1, 3, state)
		actionProbs := policy.Forward(stateTensor)

		action := SampleAction(actionProbs, epsilon)
		nextState, reward, stepDone := env.Step(action)
		done = stepDone

		// Collect data
		states = append(states, stateTensor)
		actions = append(actions, action)
		rewards = append(rewards, reward)

		state = nextState
	}

	return states, actions, rewards
}

// SampleAction chooses an action with epsilon-greedy exploration
func SampleAction(actionProbs *mat.Dense, epsilon float64) int {
	r := rand.Float64()
	if r < epsilon {
		// Exploration: choose a random action
		return rand.Intn(len(actionProbs.RawRowView(0)))
	}
	// Exploitation: choose the action with the highest probability
	return maxIndex(actionProbs.RawRowView(0))
}

func maxIndex(probs []float64) int {
	maxIdx := 0
	maxVal := probs[0]
	for i, prob := range probs {
		if prob > maxVal {
			maxVal = prob
			maxIdx = i
		}
	}
	return maxIdx
}

// ComputeLoss computes the policy gradient loss
func (pn *PolicyNetwork) ComputeLoss(states []*mat.Dense, actions []int, rewards []float64) float64 {
	normalizedRewards := normalizeRewards(rewards) // Normalize rewards for stable updates
	loss := 0.0
	for i, state := range states {
		action := actions[i]
		reward := normalizedRewards[i]
		actionProbs := pn.Forward(state)
		prob := actionProbs.At(0, action)
		if prob > 0 {
			logProb := math.Log(prob)
			loss += -logProb * reward
		}
	}
	return loss
}

// Adapt performs task-specific adaptation of the policy
func (maml *MAML) Adapt(env *NetworkEnv, initialPolicy *PolicyNetwork) *PolicyNetwork {
	adaptedPolicy := initialPolicy.Clone()

	for i := 0; i < maml.numInnerSteps; i++ {
		states, actions, rewards := SampleTrajectory(env, adaptedPolicy, 0.0) // No exploration during adaptation
		loss := adaptedPolicy.ComputeLoss(states, actions, rewards)

		// Compute gradients and apply updates
		gradFc1 := computeGradient(adaptedPolicy.fc1Weights, loss)
		gradFc2 := computeGradient(adaptedPolicy.fc2Weights, loss)
		adaptedPolicy.ApplyGradients(gradFc1, gradFc2, maml.alpha)
	}

	return adaptedPolicy
}

// MetaUpdate performs the meta-update across multiple tasks
func (maml *MAML) MetaUpdate(tasks []*NetworkEnv, epsilon float64) {
	metaLoss := 0.0
	totalReward := 0.0
	numTrajectories := 0

	gradsFc1 := mat.NewDense(maml.policy.fc1Weights.RawMatrix().Rows, maml.policy.fc1Weights.RawMatrix().Cols, nil)
	gradsFc2 := mat.NewDense(maml.policy.fc2Weights.RawMatrix().Rows, maml.policy.fc2Weights.RawMatrix().Cols, nil)

	for _, task := range tasks {
		adaptedPolicy := maml.Adapt(task, maml.policy)

		states, actions, rewards := SampleTrajectory(task, adaptedPolicy, epsilon)
		loss := adaptedPolicy.ComputeLoss(states, actions, rewards)
		metaLoss += loss

		// Accumulate rewards for average reward calculation
		for _, reward := range rewards {
			totalReward += reward
			numTrajectories++
		}

		taskGradFc1 := computeGradient(adaptedPolicy.fc1Weights, loss)
		taskGradFc2 := computeGradient(adaptedPolicy.fc2Weights, loss)

		gradsFc1.Add(gradsFc1, taskGradFc1)
		gradsFc2.Add(gradsFc2, taskGradFc2)
	}

	// Apply gradients to update the meta-policy
	maml.policy.ApplyGradients(gradsFc1, gradsFc2, maml.beta)

	// Calculate and print average reward
	averageReward := totalReward / float64(numTrajectories)
	fmt.Printf("Average Reward: %.4f\n", averageReward)
}

// ComputeGradient simplifies gradient computation
func computeGradient(weights *mat.Dense, loss float64) *mat.Dense {
	grad := mat.NewDense(weights.RawMatrix().Rows, weights.RawMatrix().Cols, nil)
	grad.Apply(func(i, j int, v float64) float64 {
		return v * loss
	}, weights)
	return grad
}

// ApplyGradients applies gradients to update the policy's parameters
func (pn *PolicyNetwork) ApplyGradients(gradFc1, gradFc2 *mat.Dense, learningRate float64) {
	updateWeights(pn.fc1Weights, gradFc1, learningRate)
	updateWeights(pn.fc2Weights, gradFc2, learningRate)
}

// Helper to update weights
func updateWeights(weights, grad *mat.Dense, learningRate float64) {
	weights.Apply(func(i, j int, v float64) float64 {
		return v - learningRate*grad.At(i, j)
	}, weights)
}

// MAML represents the meta-learning algorithm for policy optimization
type MAML struct {
	policy        *PolicyNetwork
	alpha         float64
	beta          float64
	numInnerSteps int
}

// NewMAML initializes MAML with a policy network and hyperparameters
func NewMAML(policy *PolicyNetwork, alpha, beta float64, numInnerSteps int) *MAML {
	return &MAML{
		policy:        policy,
		alpha:         alpha,
		beta:          beta,
		numInnerSteps: numInnerSteps,
	}
}

// Train runs the meta-learning loop
func (maml *MAML) Train(tasks []*NetworkEnv, numMetaIterations int) {
	epsilon := 1.0 // Initial exploration rate
	epsilonDecay := 0.995
	epsilonMin := 0.1

	for i := 0; i < numMetaIterations; i++ {
		fmt.Printf("Meta Iteration %d\n", i+1)
		maml.MetaUpdate(tasks, epsilon)
		epsilon = math.Max(epsilon*epsilonDecay, epsilonMin) // Decay epsilon
	}
}

func SampleTasks(numTasks int, data [][]float64) []*NetworkEnv {
	var tasks []*NetworkEnv
	for i := 0; i < numTasks; i++ {
		tasks = append(tasks, NewNetworkEnv(data))
	}
	return tasks
}

// Main function
func main() {
	data, err := LoadAndCleanDataset("Dataset2.csv")
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	tasks := SampleTasks(NumTasks, data)
	policy := NewPolicyNetwork(3, 128, 2)
	maml := NewMAML(policy, InnerLearningRate, OuterLearningRate, NumInnerSteps)
	maml.Train(tasks, NumMetaIterations)
}
