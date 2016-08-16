package main

import (
	"bufio"
	"fmt"
	"github.com/cousine/go-krang"
	"github.com/mgutz/ansi"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

const (
	IRIS_SETOSA     = "Iris-setosa"
	IRIS_VERSICOLOR = "Iris-versicolor"
	IRIS_VIRGINICA  = "Iris-virginica"
)

func main() {
	a_prompt := ansi.ColorFunc("yellow+bh")
	a_info := ansi.ColorFunc("cyan")
	a_success := ansi.ColorFunc("green")

	fmt.Println(a_prompt("Welcome to krang IRIS neural network test"))

	fmt.Println(a_info("Loading training samples"))
	rawData, err := os.Open("./iris.data")
	if err != nil {
		log.Fatal(err)
	}

	defer rawData.Close()

	var inputValues [][]float64
	var targetValues [][]float64

	fmt.Println(a_info("Standardizing datum"))
	scanner := bufio.NewScanner(rawData)
	for scanner.Scan() {
		columns := strings.Split(scanner.Text(), ",")
		targetTuple := columns[len(columns)-1]

		columns = columns[:len(columns)-1]

		inputValues = append(inputValues, extractInputTuple(columns))
		encodedTargetVal, _ := encodeTarget(targetTuple)
		targetValues = append(targetValues, encodedTargetVal)
	}

	inputValues = inputValues[:len(inputValues)-1]
	inputValues, minArr, maxArr := normalizeInputs(inputValues)
	targetValues = targetValues[:len(targetValues)-1]

	fmt.Println(a_success("Training data loaded and standardized"))
	fmt.Printf("%vExtracted %d samples \n%v", ansi.ColorCode("green"), len(inputValues), ansi.ColorCode("off"))
	fmt.Println(a_info("Preparing neural network"))

	topology := []uint{4, 4, 3}

	kNet := krang.NewNet(topology, 0.15, 0.5, 150.0)

	for t := 0; t < 5000; t++ {
		for i := 0; i < len(inputValues); i++ {
			kNet.FeedForward(inputValues[i])
			kNet.BackPropagate(targetValues[i])
		}
	}
	fmt.Println(a_info("Training complete"))
	fmt.Printf("Average error rate for the network: %.2f %%\n", kNet.GetRecentAverageErrorRate()*100)

	for {
		fmt.Println(a_prompt("Enter an Iris specification for recognition:"))

		var uinput string
		_, _ = fmt.Scanln(&uinput)

		input := extractInputTuple(strings.Split(uinput, ","))
		if len(input) != 4 {
			fmt.Println("Invalid input!")
			continue
		}

		kNet.FeedForward(normalizeInput(input, minArr, maxArr))

		results := kNet.GetResults()

		fmt.Printf("%.2f%% Iris-setosa, %.2f%% Iris-versicolor, %.2f%% Iris-virginica\n", results[0]*100, results[1]*100, results[2]*100)
		fmt.Println(a_prompt("Is this correct? (y/n)"))

		_, _ = fmt.Scanln(&uinput)

		switch uinput {
		case "n":
			fmt.Println(a_prompt("What was the correct classification?"))
			fmt.Println(a_prompt("1. Iris-setosa"))
			fmt.Println(a_prompt("2. Iris-versicolor"))
			fmt.Println(a_prompt("3. Iris-virginica"))

			var choice int
			_, _ = fmt.Scanf("%d", &choice)

			switch choice {
			case 1:
				uinput = IRIS_SETOSA
				break
			case 2:
				uinput = IRIS_VERSICOLOR
				break
			case 3:
				uinput = IRIS_VIRGINICA
				break
			default:
				continue
				break
			}

			correctTarget, err := encodeTarget(uinput)
			if err != 1 {
				kNet.BackPropagate(correctTarget)
				fmt.Printf("Thank you, my error rate is now: %.2f%%\n", kNet.GetRecentAverageErrorRate()*100)
			} else {
				fmt.Println(a_info("Couldn't recognize your input, skipping."))
			}
			break
		default:
			break
		}

	}
}

func extractInputTuple(inputs []string) (tuple []float64) {
	for i := 0; i < len(inputs); i++ {
		fInput, _ := strconv.ParseFloat(inputs[i], 64)
		tuple = append(tuple, fInput)
	}

	return
}

func encodeTarget(target string) (encoding []float64, err int) {
	switch target {
	case IRIS_SETOSA:
		encoding = []float64{1.0, 0.0, 0.0}
		break
	case IRIS_VERSICOLOR:
		encoding = []float64{0.0, 1.0, 0.0}
		break
	case IRIS_VIRGINICA:
		encoding = []float64{0.0, 0.0, 1.0}
		break
	default:
		err = 1
		break
	}

	return
}

func normalizeInputs(inputs [][]float64) (normalizedDatum [][]float64, minA []float64, maxA []float64) {
	minA = make([]float64, 4)
	maxA = make([]float64, 4)

	copy(minA, inputs[0])
	copy(maxA, inputs[0])

	for r := 0; r < len(inputs); r++ {
		for c := 0; c < len(inputs[r]); c++ {
			minA[c] = math.Min(inputs[r][c], minA[c])
			maxA[c] = math.Max(inputs[r][c], maxA[c])
		}
	}

	normalizedDatum = make([][]float64, len(inputs))

	for r := 0; r < len(inputs); r++ {
		normalizedDatum[r] = normalizeInput(inputs[r], minA, maxA)
	}

	return
}

func normalizeInput(input []float64, minArr []float64, maxArr []float64) (normalizedInput []float64) {
	normalizedInput = make([]float64, len(input))

	for c := 0; c < len(input); c++ {
		normalizedInput[c] = (input[c] - minArr[c]) / (maxArr[c] - minArr[c])
	}

	return
}
