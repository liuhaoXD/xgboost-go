package main

import (
	"fmt"
	"github.com/liuhaoXD/xgboost-go"
	"github.com/liuhaoXD/xgboost-go/model"
	"log"
)

func main() {
	var train model.Matrix
	for row := 0; row < 100; row++ {
		var dataRow []float32
		for col := 0; col < 3; col++ {
			dataRow = append(dataRow, float32((row+1)*(col+1)))
		}
		train = append(train, dataRow)
	}
	var trainLabels = make([]float32, 100)
	var i int
	for i < 50 {
		trainLabels[i] = float32(1 + i*i*i)
		i += 1
	}

	// convert to DMatrix
	trainMatrix, err := xgboost.DMatrixCreateFromMat(train, -1)
	if err != nil {
		log.Fatalln(err)
	}
	// load the labels
	err = trainMatrix.SetFloatInfo("label", trainLabels)
	if err != nil {
		log.Fatal(err)
	}
	// read back the labels, just a sanity check
	labels, err := trainMatrix.GetFloatInfo("label")
	if err != nil {
		log.Fatalln(err)
	}
	for i, label := range labels {
		fmt.Printf("label[%d]=%.0f\n", i, label)
	}
	// create the booster and load some parameters
	booster, err := xgboost.BoosterCreate([]*xgboost.DMatrix{trainMatrix})
	if err != nil {
		log.Fatalln(err)
	}
	err = booster.SetParam("booster", "gbtree")
	err = booster.SetParam("objective", "reg:linear")
	err = booster.SetParam("eval_metric", "error")
	err = booster.SetParam("silent", "0")
	err = booster.SetParam("max_depth", "5")
	err = booster.SetParam("eta", "0.1")
	err = booster.SetParam("min_child_weight", "1")
	err = booster.SetParam("gamma", "0.6")
	err = booster.SetParam("colsample_bytree", "1")
	err = booster.SetParam("subsample", "0.5")
	err = booster.SetParam("colsample_bytree", "1")
	err = booster.SetParam("num_parallel_tree", "1")
	err = booster.SetParam("reg_alpha", "10")

	// perform 200 learning iterations
	var iter int
	for iter < 200 {
		err = booster.UpdateOneIter(iter, trainMatrix)
		iter += 1
	}

	// predict
	var test model.Matrix
	for row := 0; row < 100; row++ {
		var dataRow []float32
		for col := 0; col < 3; col++ {
			dataRow = append(dataRow, float32((row+1)*(col+1)))
		}
		test = append(test, dataRow)
	}
	testMatrix, err := xgboost.DMatrixCreateFromMat(test, -1)
	if err != nil {
		log.Fatalln(err)
	}
	result, err := booster.Predict(testMatrix, 0, 0)
	if err != nil {
		log.Fatalln(err)
	}
	for i, v := range result {
		fmt.Printf("prediction[%d]=%.2f\n", i, v)
	}
	models, err := booster.DumpModel("", true)
	for i, model := range models {
		fmt.Printf("model[%d]=%s\n", i, model)
	}
	trainMatrix.Free()
	testMatrix.Free()
	booster.Free()
}
