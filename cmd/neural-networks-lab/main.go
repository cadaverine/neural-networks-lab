package main

import (
	"fmt"

	"github.com/cadaverine/neural-networks-lab/neural"
)

func main() {
	a := neural.Construct([]int{3, 4, 4, 4, 3})

	fmt.Println(a.String())

	a.SetInput(2, 6, 1)
	a.Recalc(neural.Functions[neural.Sigm])

	fmt.Println(a.String())
	fmt.Println(a.GetOutput())
}
