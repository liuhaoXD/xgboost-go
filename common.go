package xgboost

//#cgo LDFLAGS: -L${SRCDIR}/lib -lxgboost -lrabit -ldmlc -lstdc++ -lz -lrt -lm -lpthread -fopenmp
//#cgo CFLAGS: -I ${SRCDIR}/lib/xgboost/
//#include <stdlib.h>
//#include "c_api.h"
import "C"

import (
	"errors"
)

func checkError(res C.int) error {
	if int(res) != 0 {
		errStr := C.GoString(C.XGBGetLastError())
		if errStr == "" {
			errStr = "unknown error"
		}
		return errors.New(errStr)
	}
	return nil
}
