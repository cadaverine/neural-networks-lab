package main

import (
	"fmt"

	"github.com/cadaverine/neural-networks-lab/neural"
)

func main() {
	// a := neural.Construct([]int{10, 20, 10})
	a := neural.Construct([]int{2, 3, 1})

	fmt.Println(a.String())

	// a.SetInput(2, 6, 1, 4, 6, 8, 9, 2, 1, 4)
	a.SetInput(2, 6)
	a.Recalc(neural.Functions[neural.Sigm])

	fmt.Println(a.String())
	fmt.Println(a.GetOutput())

	fmt.Println(a.StringifyEdges())
}
