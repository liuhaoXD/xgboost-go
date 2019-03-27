package xgboost

import (
	"github.com/liuhaoXD/xgboost-go/model"
	"testing"
)

func TestCreateFromMat(t *testing.T) {
	data := [][]float32{{1, 2}, {3, 4}}

	matrix, err := DMatrixCreateFromMat(model.Matrix(data), -1)
	if err != nil {
		t.Error(err)
	}

	if matrix == nil {
		t.Error("matrix create failed")
	}

	if err = matrix.SetFloatInfo("label", []float32{123, 234}); err != nil {
		t.Error(err)
	}

	values, err := matrix.GetFloatInfo("label")
	if err != nil {
		t.Error(err)
	}

	if values[0] != 123 || values[1] != 234 {
		t.Errorf("Wrong values %v %v returned", values[0], values[1])
	}

	rowCount, err := matrix.NumRow()
	if err != nil {
		t.Error(err)
	}

	if rowCount != 2 {
		t.Error("Wrong row count returned")
	}

	colCount, err := matrix.NumCol()
	if err != nil {
		t.Error(err)
	}

	if colCount != 2 {
		t.Error("Wrong col count returned")
	}

	if err := matrix.Free(); err != nil {
		t.Error(err)
	}
}
