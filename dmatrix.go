package xgboost

//#cgo LDFLAGS: -L${SRCDIR}/lib -lxgboost -lrabit -ldmlc -lstdc++ -lz -lrt -lm -lpthread -fopenmp
//#cgo CFLAGS: -I ${SRCDIR}/lib/xgboost/
//#include <stdlib.h>
//#include "c_api.h"
import "C"

import (
	"errors"
	"github.com/liuhaoXD/xgboost-go/model"
	"reflect"
	"unsafe"
)

type DMatrix struct {
	handle C.DMatrixHandle
}

func (dMatrix *DMatrix) GetHandle() C.DMatrixHandle {
	return dMatrix.handle
}

// NumRow get number of rows.
func (dMatrix *DMatrix) NumRow() (uint32, error) {
	var count C.bst_ulong
	if err := checkError(C.XGDMatrixNumRow(dMatrix.handle, &count)); err != nil {
		return 0, err
	}

	return uint32(count), nil
}

// NumCol get number of cols.
func (dMatrix *DMatrix) NumCol() (uint32, error) {
	var count C.bst_ulong
	if err := checkError(C.XGDMatrixNumCol(dMatrix.handle, &count)); err != nil {
		return 0, err
	}

	return uint32(count), nil
}

// SetUIntInfo set uint32 vector to a content in info
func (dMatrix *DMatrix) SetUIntInfo(field string, values []uint32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetUIntInfo(dMatrix.handle, cstr, (*C.uint)(&values[0]), C.bst_ulong(len(values)))

	//runtime.KeepAlive(values)
	return checkError(res)
}

// SetGroup set label of the training matrix
func (dMatrix *DMatrix) SetGroup(group ...uint32) error {
	groupsLen := len(group)
	res := C.XGDMatrixSetGroup(dMatrix.handle, (*C.uint)(&group[0]), C.bst_ulong(groupsLen))
	//runtime.KeepAlive(group)
	return checkError(res)
}

// SetFloatInfo set float vector to a content in info
func (dMatrix *DMatrix) SetFloatInfo(field string, values []float32) error {
	cstr := C.CString(field)
	defer C.free(unsafe.Pointer(cstr))

	res := C.XGDMatrixSetFloatInfo(dMatrix.handle, cstr, (*C.float)(unsafe.Pointer(&values[0])), C.bst_ulong(len(values)))
	if err := checkError(res); err != nil {
		return err
	}
	//runtime.KeepAlive(values)

	return nil
}

// GetFloatInfo get float info vector from matrix
func (dMatrix *DMatrix) GetFloatInfo(field string) ([]float32, error) {
	fieldC := C.CString(field)
	defer C.free(unsafe.Pointer(fieldC))

	var outLenC C.bst_ulong
	var outResultPtrC *C.float

	if err := checkError(C.XGDMatrixGetFloatInfo(dMatrix.handle, fieldC, &outLenC, &outResultPtrC)); err != nil {
		return nil, err
	}

	var list []float32
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&list))
	sliceHeader.Cap = int(outLenC)
	sliceHeader.Len = int(outLenC)
	sliceHeader.Data = uintptr(unsafe.Pointer(outResultPtrC))

	n := make([]float32, len(list))
	copy(n, list)
	return n, nil
}

// GetUIntInfo get uint32 info vector from matrix
func (dMatrix *DMatrix) GetUIntInfo(field string) ([]uint32, error) {
	fieldC := C.CString(field)
	defer C.free(unsafe.Pointer(fieldC))

	var outLenC C.bst_ulong
	var outResultPtrC *C.uint

	if err := checkError(C.XGDMatrixGetUIntInfo(dMatrix.handle, fieldC, &outLenC, &outResultPtrC)); err != nil {
		return nil, err
	}

	var list []uint32
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&list)))
	sliceHeader.Cap = int(outLenC)
	sliceHeader.Len = int(outLenC)
	sliceHeader.Data = uintptr(unsafe.Pointer(outResultPtrC))

	n := make([]uint32, len(list))
	copy(n, list)
	return n, nil
}

func (dMatrix *DMatrix) Free() error {
	return checkError(C.XGDMatrixFree(dMatrix.handle))
}

func (dMatrix *DMatrix) SaveBinary(fName string, silent int) error {
	fileNameC := C.CString(fName)
	ret := C.XGDMatrixSaveBinary(dMatrix.handle, fileNameC, (C.int)(silent))
	return checkError(ret)
}

func DMatrixCreateFromFile(filename string, silent int) (*DMatrix, error) {
	fileNameC := C.CString(filename)
	silentC := C.int(silent)
	defer func() {
		C.free(unsafe.Pointer(fileNameC))
	}()
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixCreateFromFile(fileNameC, silentC, &handlerPointer)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	return &DMatrix{handlerPointer}, nil
}

func DMatrixCreateFromMat(data model.Matrix, missing float32) (*DMatrix, error) {

	// make sure all row have the same length
	rows := len(data)
	cols := len(data[0])
	for i := 0; i < rows; i++ {
		if len(data[i]) != cols {
			return nil, errors.New("inconsistent row length")
		}
	}

	dataC := make([]C.float, rows*cols)
	for i, row := range data {
		for j, v := range row {
			dataC[i*cols+j] = C.float(v)
		}
	}
	var outHandle C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat((*C.float)(unsafe.Pointer(&dataC[0])), C.bst_ulong(rows), C.bst_ulong(cols), C.float(missing), &outHandle)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	return &DMatrix{outHandle}, nil
}

func DMatrixCreateFromMatOMP(data model.Matrix, missing float32, nThread int) (*DMatrix, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, errors.New("missing data")
	}
	rows := len(data)
	cols := len(data[0])
	rowsC := C.bst_ulong(rows)
	colsC := C.bst_ulong(cols)
	missingC := C.float(missing)
	nThreadC := C.int(nThread)
	dataC := make([]C.float, rows*cols)
	for i, row := range data {
		for j, v := range row {
			dataC[i*rows+j] = C.float(v)
		}
	}
	var handlerPointer C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat_omp((*C.float)(unsafe.Pointer(&dataC[0])), rowsC, colsC, missingC, &handlerPointer, nThreadC)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	return &DMatrix{handlerPointer}, nil
}

// typedef int XGBCallbackDataIterNext( DataIterHandle data_handle, XGBCallbackSetData *set_function, DataHolderHandle set_function_handle);
func DMatrixCreateFromDataIter(cacheInfo []byte) (*DMatrix, error) {
	return nil, errors.New("DataIter not implemented yet")
}

func DMatrixCreateFromFromCSREx() (*DMatrix, error) {
	return nil, errors.New("CSREx not implemented yet")
}

func DMatrixSliceDMatrix() (*DMatrix, error) {
	return nil, errors.New("slice DMatrix not implemented yet")
}

func DMatrixCreateFromDT() (*DMatrix, error) {
	return nil, errors.New("DT not implemented yet")
}

func DMatrixCreateFromCSC() (*DMatrix, error) {
	return nil, errors.New("CSC not implemented yet")
}

func DMatrixCreateFromCSR() (*DMatrix, error) {
	return nil, errors.New("CSR not implemented yet")
}

func DMatrixCreateFromCSCEx(col int, indicex int, data []float32) (*DMatrix, error) {
	return nil, errors.New("CSCEx not implemented yet")
}

func RegisterLogCallback(callback func([]byte)) error {
	return errors.New("callback not supported yet")
}
