package main

import (
	"fmt"

	"github.com/cadaverine/neural-networks-lab/neural"
)

func main() {
	a := neural.Construct([]int{3, 4, 4, 4, 3})

	fmt.Println(a.String())
}
