package xgboost

// #cgo LDFLAGS: -L${SRCDIR}/lib -lxgboost -lrabit -ldmlc -lstdc++ -lz -lrt -lm -lpthread -fopenmp
// #cgo CFLAGS: -I ${SRCDIR}/lib/xgboost/
// #include <stdlib.h>
// #include "c_api.h"
import "C"

import (
	"unsafe"
)

type Booster struct {
	handle C.BoosterHandle
}

// DeleteParam set parameters
func (booster *Booster) DeleteParam(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	res := C.XGBoosterSetParam(booster.handle, cname, nil)
	if err := checkError(res); err != nil {
		return err
	}

	return nil
}

// UpdateOneIter update the model in one round using dtrain
func (booster *Booster) UpdateOneIter(iter int, mat *DMatrix) error {
	res := C.XGBoosterUpdateOneIter(booster.handle, C.int(iter), mat.handle)
	return checkError(res)
}

func (booster *Booster) Free() error {
	return checkError(C.XGBoosterFree(booster.handle))
}

// XGBoosterCreate creates a new booster for a given matrixes
func BoosterCreate(matrixList []*DMatrix) (*Booster, error) {
	var dMatrixHandle *C.DMatrixHandle
	handles := make([]C.DMatrixHandle, len(matrixList))
	for i, matrix := range matrixList {
		handles[i] = matrix.handle
	}
	if len(handles) > 0 {
		dMatrixHandle = (*C.DMatrixHandle)(&handles[0])
	}

	var out C.BoosterHandle
	res := C.XGBoosterCreate(dMatrixHandle, C.bst_ulong(len(handles)), &out)
	if err := checkError(res); err != nil {
		return nil, err
	}

	booster := &Booster{out}
	//runtime.SetFinalizer(booster, xdgBoosterFinalizer)

	return booster, nil
}

func (booster *Booster) SetParam(name string, value string) error {
	nameC := C.CString(name)
	valueC := C.CString(value)
	defer func() {
		C.free(unsafe.Pointer(nameC))
		C.free(unsafe.Pointer(valueC))
	}()
	ret := C.XGBoosterSetParam(booster.handle, nameC, valueC)
	return checkError(ret)
}

func (booster *Booster) BoostOneIter(dtrain *DMatrix, grad []float32, hess []float32) error {
	var (
		arrLen  = len(grad)
		arrLenC = C.bst_ulong(arrLen)
		gradC   = make([]C.float, arrLen)
		hessC   = make([]C.float, arrLen)
	)
	for i, v := range grad {
		gradC[i] = C.float(v)
	}
	for i, v := range hess {
		hessC[i] = C.float(v)
	}
	ret := C.XGBoosterBoostOneIter(booster.handle, dtrain.handle, (*C.float)(unsafe.Pointer(&gradC[0])), (*C.float)(unsafe.Pointer(&hessC[0])), arrLenC)
	return checkError(ret)
}

func (booster *Booster) EvalOneIter(iter int, dmats []*DMatrix, evnames []string) (result string, err error) {
	var (
		dmatsLen  = len(dmats)
		dmatsLenC = C.bst_ulong(dmatsLen)
		evnamesC  = make([]*C.char, len(evnames))
		resultC   *C.char
	)
	handles := make([]C.DMatrixHandle, dmatsLen)
	for i, dmat := range dmats {
		handles[i] = dmat.handle
	}
	for i, v := range evnames {
		evnamesC[i] = C.CString(v)
	}
	defer func() {
		for _, v := range evnamesC {
			C.free(unsafe.Pointer(v))
		}
	}()
	ret := C.XGBoosterEvalOneIter(booster.handle, C.int(iter), (*C.DMatrixHandle)(unsafe.Pointer(handles[0])), (**C.char)(unsafe.Pointer(&evnamesC[0])), dmatsLenC, (**C.char)(unsafe.Pointer(&resultC)))
	if err := checkError(ret); err != nil {
		return "", err
	}
	result = C.GoString(resultC)
	return result, nil
}
func (booster *Booster) Predict(dMatrix *DMatrix, optionMask int, ntreeLimit uint) (result []float32, err error) {
	var (
		outPtr *C.float
		outLen C.bst_ulong
	)
	ret := C.XGBoosterPredict(booster.handle, dMatrix.handle, C.int(optionMask), C.unsigned(ntreeLimit), &outLen, &outPtr)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, float32(*p))
		p = (*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

// LoadModel load model from existing file
func (booster *Booster) LoadModel(fname string) error {
	fnameC := C.CString(fname)
	defer func() {
		C.free(unsafe.Pointer(fnameC))
	}()
	ret := C.XGBoosterLoadModel(booster.handle, fnameC)
	return checkError(ret)
}

// SaveModel save model into file
func (booster *Booster) SaveModel(fname string) error {
	fNameC := C.CString(fname)
	defer func() {
		C.free(unsafe.Pointer(fNameC))
	}()
	ret := C.XGBoosterSaveModel(booster.handle, fNameC)
	return checkError(ret)
}

func (booster *Booster) GetModelRaw() (buf []byte, err error) {
	var (
		resultC *C.char
		outLen  C.bst_ulong
	)
	ret := C.XGBoosterGetModelRaw(booster.handle, &outLen, &resultC)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	buf = C.GoBytes(unsafe.Pointer(resultC), C.int(outLen))
	return buf, nil
}

func (booster *Booster) DumpModel(fMap string, withStats bool) (result []string, err error) {
	fMapC := C.CString(fMap)
	defer C.free(unsafe.Pointer(fMapC))

	var (
		outPtr     **C.char
		outLen     C.bst_ulong
		withStatsC C.int
	)
	if withStats {
		withStatsC = 1
	} else {
		withStatsC = 0
	}
	if err := checkError(C.XGBoosterDumpModel(booster.handle, fMapC, withStatsC, &outLen, &outPtr)); err != nil {
		return nil, err
	}

	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

func (booster *Booster) DumpModelEx(fMap string, withStats bool, format string) (result []string, err error) {
	fMapC := C.CString(fMap)
	formatC := C.CString(format)
	defer func() {
		C.free(unsafe.Pointer(fMapC))
		C.free(unsafe.Pointer(formatC))
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	if err := checkError(C.XGBoosterDumpModelEx(booster.handle, fMapC, withStatsC, formatC, &outLen, &outPtr)); err != nil {
		return nil, err
	}

	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

func (booster *Booster) DumpModelWithFeatures(fNum int, fNames []string, ftypes []string, withStats bool) (result []string, err error) {
	fnameC := make([]*C.char, len(fNames))
	ftypeC := make([]*C.char, len(ftypes))
	for i, v := range fNames {
		fnameC[i] = C.CString(v)
	}
	for i, v := range ftypes {
		ftypeC[i] = C.CString(v)
	}
	defer func() {
		for _, v := range fnameC {
			C.free(unsafe.Pointer(v))
		}
		for _, v := range ftypeC {
			C.free(unsafe.Pointer(v))
		}
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModelWithFeatures(booster.handle, C.int(fNum), (**C.char)(unsafe.Pointer(&fnameC[0])), (**C.char)(unsafe.Pointer(&ftypeC[0])), withStatsC, &outLen, &outPtr)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

func (booster *Booster) DumpModelExWithFeatures(fNum int, fNames []string, ftypes []string, withStats bool, format string) (result []string, err error) {
	formatC := C.CString(format)
	fnameC := make([]*C.char, len(fNames))
	ftypeC := make([]*C.char, len(ftypes))
	for i, v := range fNames {
		fnameC[i] = C.CString(v)
	}
	for i, v := range ftypes {
		ftypeC[i] = C.CString(v)
	}
	defer func() {
		C.free(unsafe.Pointer(formatC))
		for _, v := range fnameC {
			C.free(unsafe.Pointer(v))
		}
		for _, v := range ftypeC {
			C.free(unsafe.Pointer(v))
		}
	}()
	var withStatsC C.int
	if withStats {
		withStatsC = 1
	}
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterDumpModelExWithFeatures(booster.handle, C.int(fNum), (**C.char)(unsafe.Pointer(&fnameC[0])), (**C.char)(unsafe.Pointer(&ftypeC[0])), withStatsC, formatC, &outLen, &outPtr)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

func (booster *Booster) GetAttr(key string) (out string, err error) {
	keyC := C.CString(key)
	defer C.free(unsafe.Pointer(keyC))
	var (
		outC     *C.char
		successC C.int
	)
	ret := C.XGBoosterGetAttr(booster.handle, keyC, (**C.char)(unsafe.Pointer(&outC)), &successC)
	if err := checkError(ret); err != nil {
		return "", err
	}
	out = C.GoString(outC)
	return out, nil
}

func (booster *Booster) SetAttr(key string, value string) error {
	keyC := C.CString(key)
	valueC := C.CString(value)
	defer func() {
		C.free(unsafe.Pointer(keyC))
		C.free(unsafe.Pointer(valueC))
	}()
	ret := C.XGBoosterSetAttr(booster.handle, keyC, valueC)
	return checkError(ret)
}

func (booster *Booster) GetAttrNames() (result []string, err error) {
	var (
		outPtr **C.char
		outLen C.bst_ulong
	)
	ret := C.XGBoosterGetAttrNames(booster.handle, &outLen, &outPtr)
	if err := checkError(ret); err != nil {
		return nil, err
	}
	p := outPtr
	ptrSize := unsafe.Sizeof(*p)
	var i C.bst_ulong
	for i < outLen {
		result = append(result, C.GoString(*p))
		p = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(ptrSize)))
		i += 1
	}
	return result, nil
}

func (booster *Booster) LoadModelFromBuffer(buffer []byte) error {
	ret := C.XGBoosterLoadModelFromBuffer(booster.handle, (unsafe.Pointer)(&buffer[0]), (C.bst_ulong)(len(buffer)))
	return checkError(ret)
}

func (booster *Booster) LoadRabitCheckpoint(version int) error {
	ret := C.XGBoosterLoadRabitCheckpoint(booster.handle, (*C.int)((unsafe.Pointer)(&version)))
	return checkError(ret)
}

func (booster *Booster) SaveRabitCheckpoint() error {
	ret := C.XGBoosterSaveRabitCheckpoint(booster.handle)
	return checkError(ret)
}
