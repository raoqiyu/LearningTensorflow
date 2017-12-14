## Swig

python 调用C函数，通常会生成pywrap函数

tensorflow/python/eager/pywrap_*文件对应的函数便是对tensorflow/c/eager/进行封装

对应的swig文件为tensorflow/python/pywrap_tfe.i



Swig/basic_example 是取自[官网](http://www.swig.org/Doc3.0/Python.html#Python_nn4)的例子
Swig/return values from arguments 将上面的例子改为通过指针参数进行多参数的返回，在tensorflow的实现中也用到了这种方法