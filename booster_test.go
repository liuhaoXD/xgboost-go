package xgboost

import (
	"io/ioutil"
	"math"
	"os"
	"path"
)
import "testing"

func TestBooster(t *testing.T) {
	// create the training data
	cols := 3
	rows := 5
	trainData := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		for j := 0; j < cols; j++ {
			row[j] = float32((i + 1) * (j + 1))
		}
		trainData[i] = row
	}

	trainLabels := make([]float32, rows)
	for i := 0; i < rows; i++ {
		trainLabels[i] = float32(1 + i*i*i)
	}

	matrix, err := DMatrixCreateFromMat(trainData, -1)
	if err != nil {
		t.Error(err)
	}

	err = matrix.SetFloatInfo("label", trainLabels)
	if err != nil {
		t.Error(err)
	}

	booster, err := BoosterCreate([]*DMatrix{matrix})
	if err != nil {
		t.Error(err)
	}

	noErr := func(err error) {
		if err != nil {
			t.Error(err)
		}
	}

	noErr(booster.SetParam("booster", "gbtree"))
	noErr(booster.SetParam("objective", "reg:linear"))
	noErr(booster.SetParam("max_depth", "5"))
	noErr(booster.SetParam("eta", "0.1"))
	noErr(booster.SetParam("min_child_weight", "1"))
	noErr(booster.SetParam("subsample", "0.5"))
	noErr(booster.SetParam("colsample_bytree", "1"))
	noErr(booster.SetParam("num_parallel_tree", "1"))
	noErr(booster.SetParam("silent", "1"))

	// perform 200 learning iterations
	for iter := 0; iter < 200; iter++ {
		noErr(booster.UpdateOneIter(iter, matrix))
	}

	testData := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		for j := 0; j < cols; j++ {
			row[j] = float32((i + 1) * (j + 1))
		}
		testData[i] = row
	}

	testmat, err := DMatrixCreateFromMat(testData, -1)
	if err != nil {
		t.Error(err)
	}

	res, err := booster.Predict(testmat, 0, 0)
	if err != nil {
		t.Error(err)
	}

	// TODO measure actual accuracy
	totalDiff := 0.0
	for i, label := range trainLabels {
		diff := math.Abs(float64(label - res[i]))
		totalDiff += diff
	}

	if totalDiff > 6.0 {
		t.Error("error is too large")
	}

	dir, err := ioutil.TempDir("", "go-xgboost")
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(dir) // clean up

	savePath := path.Join(dir, "testmodel.bst")

	noErr(booster.SaveModel(savePath))

	newBooster, err := BoosterCreate(nil)
	if err != nil {
		t.Error(err)
	}

	noErr(newBooster.LoadModel(savePath))

	testmat2, err := DMatrixCreateFromMat(testData, -1)
	if err != nil {
		t.Error(err)
	}

	res, err = newBooster.Predict(testmat2, 0, 0)
	if err != nil {
		t.Error(err)
	}

	// TODO measure actual accuracy
	totalDiff = 0.0
	for i, label := range trainLabels {
		diff := math.Abs(float64(label - res[i]))
		totalDiff += diff
	}

	if totalDiff > 6.0 {
		t.Error("error is too large")
	}
}
