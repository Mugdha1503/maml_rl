package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Hyperparameters
const (
	InnerLearningRate  = 0.01  // Alpha
	OuterLearningRate  = 0.001 // Beta
	NumInnerSteps      = 5     // Number of inner adaptation steps
	NumMetaIterations  = 50    // Number of meta-training iterations
	MaxReward          = 100    // Max reward
	MinReward          = -100   // Min reward
)

// PolicyNetwork defines the structure of the neural network
type PolicyNetwork struct {
	Weight1, Bias1, Weight2, Bias2 *G.Node
}

// NetworkEnv represents the environment for network tasks
type NetworkEnv struct {
	Path1RTT, Path1CWND, Path1Inflight float64
	Path2RTT, Path2CWND, Path2Inflight float64
	Reward                             float64
}

func InitializePolicyNetwork(g *G.ExprGraph) *PolicyNetwork {
	weight1 := G.NewMatrix(g, tensor.Float64, G.WithShape(6, 128), G.WithName("W1"), G.WithInit(G.GlorotU(1)))
	bias1 := G.NewVector(g, tensor.Float64, G.WithShape(128), G.WithName("b1"), G.WithInit(G.Zeroes()))
	weight2 := G.NewMatrix(g, tensor.Float64, G.WithShape(128, 2), G.WithName("W2"), G.WithInit(G.GlorotU(1)))
	bias2 := G.NewVector(g, tensor.Float64, G.WithShape(2), G.WithName("b2"), G.WithInit(G.Zeroes()))

	return &PolicyNetwork{
		Weight1: weight1,
		Bias1:   bias1,
		Weight2: weight2,
		Bias2:   bias2,
	}
}

// Forward pass of the policy network
func (p *PolicyNetwork) Forward(state *G.Node) *G.Node {
	layer1 := G.Must(G.Add(G.Must(G.Mul(state, p.Weight1)), p.Bias1))
	layer1Activ := G.Must(G.Rectify(layer1))
	outputLayer := G.Must(G.Add(G.Must(G.Mul(layer1Activ, p.Weight2)), p.Bias2))
	return G.Must(G.SoftMax(outputLayer))
}

// Load and clean the dataset from the CSV file
func LoadAndCleanDataset(filename string) ([]NetworkEnv, error) {
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

	if len(records) < 2 {
		return nil, fmt.Errorf("not enough data in the CSV file")
	}

	var environments []NetworkEnv

	cleanColumn := func(column []string) ([]float64, error) {
		var cleaned []float64
		var sum float64
		var count int

		for _, value := range column {
			num, err := strconv.ParseFloat(value, 64)
			if err != nil || math.IsNaN(num) {
				num = math.NaN()
			} else {
				sum += num
				count++
			}
			cleaned = append(cleaned, num)
		}

		if count == 0 {
			return cleaned, fmt.Errorf("no valid numbers in column")
		}

		mean := sum / float64(count)
		for i := range cleaned {
			if math.IsNaN(cleaned[i]) {
				cleaned[i] = mean
			}
		}
		return cleaned, nil
	}

	normalize := func(column []float64) []float64 {
		min := column[0]
		max := column[0]
		for _, v := range column {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		for i := range column {
			column[i] = (column[i] - min) / (max - min)
		}
		return column
	}

	columns := make([][]float64, 6)
	for i := 0; i < 6; i++ {
		col, err := cleanColumn(getColumn(records, i))
		if err != nil {
			return nil, err
		}
		columns[i] = normalize(col)
	}

	for i := 1; i < len(records); i++ { // Skip header
		env := NetworkEnv{
			Path1RTT:      columns[0][i-1],
			Path1CWND:     columns[1][i-1],
			Path1Inflight: columns[2][i-1],
			Path2RTT:      columns[3][i-1],
			Path2CWND:     columns[4][i-1],
			Path2Inflight: columns[5][i-1],
		}
		environments = append(environments, env)
	}
	return environments, nil
}

// Get a column from a CSV by index
func getColumn(records [][]string, index int) []string {
	if index < 0 || index >= len(records[0]) {
		return nil // Return nil for invalid index
	}
	column := []string{}
	for _, record := range records {
		if index < len(record) {
			column = append(column, record[index])
		}
	}
	return column
}

// Reward calculation using weights and action
func calculateReward(path1 []float64, path2 []float64, action int) float64 {
	weights := []float64{0.5, 0.3, 0.2}
	score1 := 0.0
	score2 := 0.0
	for i := range path1 {
		score1 += weights[i] * path1[i]
		score2 += weights[i] * path2[i]
	}
	var reward float64
	if action == 0 {
		reward = score2 - score1
	} else {
		reward = score1 - score2
	}
	return math.Max(MinReward, math.Min(reward, MaxReward))
}

func CollectTrajectories(env *NetworkEnv, policy *PolicyNetwork, g *G.ExprGraph) (float64, error) {
	// Create a Gorgonia tensor from the environment state
	stateTensor := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, 6), tensor.WithBacking([]float64{
		env.Path1RTT,
		env.Path1CWND,
		env.Path1Inflight,
		env.Path2RTT,
		env.Path2CWND,
		env.Path2Inflight,
	}))

	// Create a Gorgonia node for the state
	state := G.NewTensor(g, tensor.Float64, 2, G.WithShape(1, 6), G.WithValue(stateTensor))

	// Forward pass through the policy network
	actionProbabilities := policy.Forward(state)

	// Create a VM to run the graph
	vm := G.NewTapeMachine(g)

	// Run the graph to compute action probabilities
	if err := vm.RunAll(); err != nil {
		return 0, fmt.Errorf("failed to run the graph: %v", err)
	}
	defer vm.Reset()

	// Extract tensor data from action probabilities
	actionProbsTensor := actionProbabilities.Value().(tensor.Tensor)
	actionProbsData := actionProbsTensor.Data().([]float64)

	// Simulate action selection
	action := 0
	sample := rand.Float64()
	if sample < actionProbsData[0] {
		action = 0
	} else {
		action = 1
	}

	// Calculate reward based on the selected action
	path1 := []float64{env.Path1RTT, env.Path1CWND, env.Path1Inflight}
	path2 := []float64{env.Path2RTT, env.Path2CWND, env.Path2Inflight}
	reward := calculateReward(path1, path2, action)
	env.Reward = reward

	return reward, nil
}

func ComputeLoss(g *G.ExprGraph, reward float64, actionProbabilities *G.Node) *G.Node {
	// Compute the log probabilities
	logProbabilities, err := G.Log(actionProbabilities)
	if err != nil {
		log.Fatalf("Failed to compute log probabilities: %v", err)
	}

	// We want to sum the log probabilities for the selected action to get a scalar
	sumLogProbabilities := G.Must(G.Sum(logProbabilities))

	// The final loss is the negative of the reward multiplied by the sum of the log probabilities
	return G.Must(G.Neg(G.Must(G.Mul(G.NewConstant(reward), sumLogProbabilities))))
}

func MetaTrainingLoop(g *G.ExprGraph, policy *PolicyNetwork, environments []NetworkEnv) {
	for metaIteration := 0; metaIteration < NumMetaIterations; metaIteration++ {
		// Task-specific adaptation
		for _, task := range environments {
			// Collect trajectories for the task
			averageReward, err := CollectTrajectories(&task, policy, g)
			if err != nil {
				log.Fatalf("Failed to collect trajectories: %v", err)
			}

			// Calculate action probabilities using the policy
			stateTensor := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, 6), tensor.WithBacking([]float64{
				task.Path1RTT,
				task.Path1CWND,
				task.Path1Inflight,
				task.Path2RTT,
				task.Path2CWND,
				task.Path2Inflight,
			}))

			state := G.NewTensor(g, tensor.Float64, 2, G.WithShape(1, 6), G.WithValue(stateTensor))

			actionProbabilities := policy.Forward(state)

			// Compute loss
			loss := ComputeLoss(g, averageReward, actionProbabilities)

			// Ensure that the forward pass has been run before applying the gradients
			vm := G.NewTapeMachine(g)
			if err := vm.RunAll(); err != nil {
				log.Fatalf("Failed to run forward pass: %v", err)
			}
			defer vm.Close()

			// Compute gradients and update parameters
			grads, err := G.Grad(loss, policy.Weight1, policy.Bias1, policy.Weight2, policy.Bias2)
			if err != nil {
				log.Fatalf("Failed to compute gradients: %v", err)
			}

			// Update task-specific parameters using the gradients
			for i, param := range []*G.Node{policy.Weight1, policy.Bias1, policy.Weight2, policy.Bias2} {
				paramUpdated := G.Must(G.Sub(param, G.Must(G.Mul(grads[i], G.NewConstant(InnerLearningRate)))))

				// Update the parameter in the policy network
				switch i {
				case 0:
					policy.Weight1 = paramUpdated
				case 1:
					policy.Bias1 = paramUpdated
				case 2:
					policy.Weight2 = paramUpdated
				case 3:
					policy.Bias2 = paramUpdated
				}
			}
		}

		// Meta-update: Sample new trajectories using adapted parameters
		metaLoss := G.NewScalar(g, tensor.Float64, G.WithName("meta_loss"), G.WithValue(0.0))
		for _, task := range environments {
			averageReward, err := CollectTrajectories(&task, policy, g)
			if err != nil {
				log.Fatalf("Failed to collect trajectories: %v", err)
			}

			// Calculate action probabilities again after task adaptation
			state := G.NewMatrix(g, tensor.Float64, G.WithShape(1, 6),
				G.WithValue([]float64{
					float64(task.Path1RTT),
					float64(task.Path1CWND),
					float64(task.Path1Inflight),
					float64(task.Path2RTT),
					float64(task.Path2CWND),
					float64(task.Path2Inflight),
				}))

			actionProbabilities := policy.Forward(state)

			// Compute meta-loss
			metaLoss = G.Must(G.Add(metaLoss, ComputeLoss(g, averageReward, actionProbabilities)))
		}

		// Update policy parameters using the accumulated meta-loss
		metaGrads, err := G.Grad(metaLoss, policy.Weight1, policy.Bias1, policy.Weight2, policy.Bias2)
		if err != nil {
			log.Fatalf("Failed to compute meta-gradients: %v", err)
		}

		for i, param := range []*G.Node{policy.Weight1, policy.Bias1, policy.Weight2, policy.Bias2} {
			paramUpdated := G.Must(G.Sub(param, G.Must(G.Mul(metaGrads[i], G.NewConstant(OuterLearningRate)))))

			// Update the parameter in the policy network
			switch i {
			case 0:
				policy.Weight1 = paramUpdated
			case 1:
				policy.Bias1 = paramUpdated
			case 2:
				policy.Weight2 = paramUpdated
			case 3:
				policy.Bias2 = paramUpdated
			}
		}

		// Print progress
		if metaIteration%10 == 0 {
			fmt.Printf("Meta Iteration: %d\n", metaIteration)
		}
	}
}

func main() {
	rand.NewSource(time.Now().UnixNano())

	// Initialize Gorgonia graph
	g := G.NewGraph()

	// Load the dataset
	environments, err := LoadAndCleanDataset("maml/Dataset2.csv")
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	// Initialize the policy network
	policy := InitializePolicyNetwork(g)

	// Start meta-training loop
	MetaTrainingLoop(g, policy, environments)
}
