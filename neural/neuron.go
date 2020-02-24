package neural

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

var Functions = map[Function]func(float64) float64{
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
	return fmt.Sprint(n.GetValue(Functions[Sigm]))
}
