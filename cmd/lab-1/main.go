package main

import (
	"fmt"
	"math"
)

// Function - типы функций акцивации
type Function int

const (
	// Sigm - сигмоидальная функция
	Sigm Function = iota
	// Tahm - гиперболический тангенс
	Tahm
)

var functions = map[Function]func(float64) float64{
	Sigm: func(x float64) float64 { return 1 / (1 + math.Pow(math.E, -x)) },
	Tahm: func(x float64) float64 { return (math.Pow(math.E, 2*x) - 1) / (math.Pow(math.E, 2*x) + 1) },
}

// Neuron - нейрон
type Neuron struct {
	Sum  float64
	Bias float64
}

// GetValue - получение значения нейрона
// (возможно передать собственную функцию активации)
func (n *Neuron) GetValue(f func(float64) float64) float64 {
	return f(n.Sum + n.Bias)
}

func (n Neuron) String() string {
	return fmt.Sprint(n.GetValue(functions[Sigm]))
}

// Edge - связь между нейронами
type Edge struct {
	From   *Neuron
	To     *Neuron
	Weight float64
}

// EdgeLayer - слой связей нейронной сети (список ребер)
type EdgeLayer []*Edge

// NeuronLayer - слой нейронов сети
type NeuronLayer []*Neuron

// NeuralNetwork - нейронная сеть (список слоев)
type NeuralNetwork struct {
	nLayers []NeuronLayer
	eLayers []EdgeLayer
}

// recalc - пересчитать значения нейронов
func (nn *NeuralNetwork) recalc(f func(float64) float64) {
	var renewed map[*Neuron]struct{}

	for _, layer := range nn.eLayers {
		for _, edge := range layer {
			if _, ok := renewed[edge.To]; !ok {
				edge.To.Sum = 0
				renewed[edge.To] = struct{}{}
			}

			edge.To.Sum += edge.Weight * edge.From.GetValue(f)
		}
	}
}

func (nn NeuralNetwork) String() string {
	var result string

	for i, layer := range nn.nLayers {
		result += fmt.Sprintf("\nLayer %v:\n", i+1)

		for j, n := range layer {
			result += n.String()

			if j < len(layer)-1 {
				result += ", "
			}
		}

	}

	return result
}

// [3, 4, 4, 4, 3]
func construct(nums []int) *NeuralNetwork {
	var nn NeuralNetwork

	for i, num := range nums {
		var neurons NeuronLayer

		for j := 0; j < num; j++ {
			neurons = append(neurons, &Neuron{})
		}

		nn.nLayers = append(nn.nLayers, neurons)

		if i > 0 {
			var edges EdgeLayer

			for _, from := range nn.nLayers[i-1] {
				for _, to := range nn.nLayers[i] {
					edges = append(edges, &Edge{from, to, 1})
				}
			}

			nn.eLayers = append(nn.eLayers, edges)
		}
	}

	return &nn
}

func main() {
	a := construct([]int{2, 3, 2})

	fmt.Println(a.String())
}
