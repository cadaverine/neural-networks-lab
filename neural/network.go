package neural

import (
	"fmt"

	"github.com/pkg/errors"
)

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

// Network - нейронная сеть (список слоев)
type Network struct {
	nLayers []NeuronLayer
	eLayers []EdgeLayer
}

// StringifyEdges - распечатать связи
func (n *Network) StringifyEdges() string {
	var result string

	for i, layer := range n.eLayers {
		result += fmt.Sprintf("edges layer %v:\n", i)

		for j, edge := range layer {
			result += fmt.Sprintf("%v: %v\n", j, edge)
		}
	}

	return result
}

// Recalc - пересчитать значения (суммы) нейронов
func (n *Network) Recalc(f func(float64) float64) {
	renewed := make(map[*Neuron]struct{})

	for _, layer := range n.eLayers {
		for _, edge := range layer {
			if _, ok := renewed[edge.To]; !ok {
				edge.To.Sum = 0
				renewed[edge.To] = struct{}{}
			}

			edge.To.Sum += edge.Weight * edge.From.GetValue(f)
		}
	}
}

// Construct - конструктор сети (пример входных параметров: [3, 4, 4, 4, 3])
func Construct(nums []int) *Network {
	var n Network

	for i, num := range nums {
		var neurons NeuronLayer

		for j := 0; j < num; j++ {
			neurons = append(neurons, &Neuron{})
		}

		n.nLayers = append(n.nLayers, neurons)

		if i > 0 {
			var edges EdgeLayer

			for _, from := range n.nLayers[i-1] {
				for _, to := range n.nLayers[i] {
					edges = append(edges, &Edge{from, to, 1})
				}
			}

			n.eLayers = append(n.eLayers, edges)
		}
	}

	return &n
}

func (n *Network) String() string {
	var result string

	for i, layer := range n.nLayers {
		result += fmt.Sprintf("layer %v: ", i+1)

		for j, n := range layer {
			result += n.String()

			if j < len(layer)-1 {
				result += ", "
			} else {
				result += ";\n"
			}
		}
	}

	return result
}

func (n *Network) GetInputLayer() NeuronLayer {
	return n.nLayers[0]
}

func (n *Network) GetOutputLayer() NeuronLayer {
	return n.nLayers[len(n.nLayers)-1]
}

// SetInput - задать значения входного слоя
func (n *Network) SetInput(values ...float64) error {
	inputLayer := n.GetInputLayer()

	if len(values) != len(inputLayer) {
		return errors.New("values number must be equal to input layer length")
	}

	for i, value := range values {
		inputLayer[i].Sum = value
	}

	return nil
}

// GetOutput - получить значения выходного слоя
func (n *Network) GetOutput() []float64 {
	var output []float64

	for _, neuron := range n.GetOutputLayer() {
		output = append(output, neuron.GetValue(Functions[Sigm]))
	}

	return output
}

// Learning - обучение нейроннной сети
func (n *Network) Learning(inputs []float64, outputs []float64) error {
	if len(inputs) != len(n.GetInputLayer()) {
		return errors.New("inputs number must be equal to input layer length")
	}

	if len(outputs) != len(n.GetOutputLayer()) {
		return errors.New("outputs number must be equal to output layer length")
	}

	return nil
}
