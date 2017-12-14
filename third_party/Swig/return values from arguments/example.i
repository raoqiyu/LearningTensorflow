/* File: example.i */
%module example

%{
#define SWIG_FILE_WITH_INIT
#include "example.h"
%}

/* Set the input argument to point to a temporary variable */
%typemap(in, numinputs=0) int *a (int temp) {
   $1 = &temp;
}

%typemap(argout) int *a {
   // Append output value $1 to $result
   PyObject *o, *o2, *o3;
   o = PyFloat_FromDouble(*$1);
   if ((!$result) || ($result == Py_None)) {
       $result = o;
   } else {
       if (!PyTuple_Check($result)) {
           PyObject *o2 = $result;
           $result = PyTuple_New(1);
           PyTuple_SetItem($result,0,o2);
       }
       o3 = PyTuple_New(1);
       PyTuple_SetItem(o3,0,o);
       o2 = $result;
       $result = PySequence_Concat(o2,o3);
       Py_DECREF(o2);
       Py_DECREF(o3);
   }

}
int fact(int *a,int n);
