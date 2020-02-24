package neural

import "fmt"

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

// recalc - пересчитать значения (суммы) нейронов
func (nn *Network) recalc(f func(float64) float64) {
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

// Construct - конструктор сети (пример входных параметров: [3, 4, 4, 4, 3])
func Construct(nums []int) *Network {
	var nn Network

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

func (nn *Network) String() string {
	var result string

	for i, layer := range nn.nLayers {
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
