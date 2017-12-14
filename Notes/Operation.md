## Tensorflow Operation
tensorflow中operation的实现，主要涉及以下几个目录
```
├── tensorflow
│   ├── core
│   │   ├── framework
│   │   ├── kernels
│   │   ├── ops
│   │   ├── user_ops
│   ├── python
│   │   ├── eager
│   │   ├── framework
│   │   ├── ops
│   │   ├── user_ops
│   └── user_ops
```

#### tensorflow/core 主要涉及具体的底层实现

    - tensorflow/core/kernel: Op的具体实现，CPU版和GPU版， 并调用REGISTER_KERNEL_BUILDER宏对Op进行注册
        同一个Op可以注册多次（CPU，GPU）,一个Op一般对应三个文件：Op.h Op.cc Op.cu.cc
        Op.h     : 定义operation的template
        Op.cc    : 实现operation的CPU版本，并调用REGISTER_KERNEL_BUILDER注册该Op的所有版本，包括GPU版
        Op.cu.cc : 实现operation的GPU版本（采用独立的cu.cc文件，方便CUDA编译）

        当Op需要应对多种不同类型的参数，可以采用宏TF_CALL_REAL_NUMBER_TYPES，TF_CALL_GPU_NUMBER_TYPES，
        等对需要的类型进行注册，此时可能存在多个cc文件实现不同版本的Op

    - tensorflow/core/ops:  Op属性的注册，kernel中具体实现了各种各样的Op。
        ops将Op分类例如array_op, nn_op, random_op等，然后采用宏REGISTER_OP对注册Op的interface，包括Op的：
        name, inputs,(names and types), outputs(names and types)和docstring，以及各种attrs。还有ShapeFn，
        用来assert inputs的shape适合graph的构造，以及设定outputs的shape

        这里也会对计算Op gradient的函数进行注册，具体采用FunctionDefHelper生成函数定义，采用REGISTER_OP_GRADIENT
        对函数进行注册，梯度计算函数的具体实现在pytho/ops下进行并注册到tensorflow中

    - tensorflow/core/framework: 上述用到的REGISTER_KERNEL_BUILDER，TF_CALL_REAL_NUMBER_TYPES，
        FunctionDefHelper，REGISTER_OP_GRADIENT等都是定义在这个目录中


### tensorflow/python 主要涉及python版API的实现

tensorflow编译完成后，会将所有的Op打包进tensorflow/python/_pywrap_tensorflow_internal.so这个文件中

在python 中import tensorflow时会通过tensorflow/python/pywrap_tensorflow.py将该so文件加载到内存
(该操作在tensorflow/python/__init__.py中进行)



    - tensorflow/python/ops: 该目录下的文件主要是为例调用tensorflow/core/ops对应的Op，包含三种文件
        *_op.py         :  由python直接调用
        *_gradient.py   :  计算相应Op的Gradient，并将函数注册到tensorflow
        gen_*_op.py     :  编译生成的文件，对*_op.py的调用会转为对这个文件相应Op的调用，然后通过
                            pywrap_tensorflow.TFE_Py_Execute调用 _pywrap_tensorflow_internal.so中
                            打包好的函数

    - tensorflow/python/eager: 定义了python与低层通信的函数？？？，如TFE_Py_Execute

[user_ops](https://www.tensorflow.org/extend/adding_an_op)对应的是用户自定义的函数


---

例子：
tf.reduce_sum的调用过程
```
  tf.reduce_sum 在python/ops/math_ops.py中
    -> gen_math_ops._sum()
        -> _execute.execute(b"Sum" ,*)
            -> pywrap_tensorflow.TFE_Py_Execute() #pywrap_tensorflow对应的是_pywrap_tensorflow_internal.so
```
tf.reduce_sum的实现，上诉调用过程中，发现最后调用的应该是kernel中Sum函数

 tensorflow/core/kernel下与Sum函数相关的代码有
```
reduction_gpu_kernels.cu.h
reduction_ops.h
reduction_ops_all.cc
reduction_ops_any.cc
reduction_ops_common.cc
reduction_ops_common.h
reduction_ops_gpu_bool.cu.cc
reduction_ops_gpu_complex64.cu.cc
reduction_ops_gpu_complex128.cu.cc
reduction_ops_gpu_double.cu.cc
reduction_ops_gpu_float.cu.cc
reduction_ops_gpu_int.cu.cc
reduction_ops_half_mean_sum.cu.cc
reduction_ops_sum.cc
 ```
 这里将```reduce_sum,reduce_max,reduce_min,reduce_mean,reduce_prod```写在一起，所以代码较复杂，

 下面四个文件定义reduce_*函数的接口

```reduction_ops.h
reduction_ops_all.cc
reduction_ops_any.cc
reduction_ops_common.cc
reduction_ops_common.h
```

```
reduction_ops_sum.cc            实现了CPU版的Sum函数,并采用TF_CALL_NUMBER_TYPES对各种参数类型进行注册
reduction_gpu_kernels.cu.h      实现了GPU版的Sum函数
下面几个文件主要对应不同参数类型的实现，bool,complex64,complex128,double,int half
reduction_ops_gpu_bool.cu.cc
reduction_ops_gpu_complex64.cu.cc
reduction_ops_gpu_complex128.cu.cc
reduction_ops_gpu_double.cu.cc
reduction_ops_gpu_float.cu.cc
reduction_ops_gpu_int.cu.cc
reduction_ops_half_mean_sum.cu.cc

```


---
**然而**上诉调用过程虽然直观，能够看出python operation与从C++ operation的对应关系，但是实际的执行路径不是这样的。
在gen_*_ops.py中，每个operation有两个执行路径，实际的执行流程```_op_def_lib._apply_op_helper```
```
_, _, _op = _op_def_lib._apply_op_helper(
        "Op name", input=input, reduction_indices=reduction_indices,
        keep_dims=keep_dims, name=name)
```
在每个gen_*_ops.py文件被import时，将该文件中operation的描述```Op_def```添加到```_op_def_lib._ops```（该操作在gen_*_ops.py的最后一行）。
```_op_def_lib._ops```是一个dict，其中存放<'op_name':op_info在内存中的位置>,例如：

```
'Sum': <tensorflow.python.framework.op_def_library._OpInfo at 0x7f6ff55b3400>
```
op_def 的定义在 ```tensorflow/core/framework/op_def.proto```中

因此，实际的执行流程是


```
import tensorflow  #会将operation的Op_def加载进内存
tf.reduce_sum # 调用这个函数，不会实际执行，而是将该op添加到当前的graph中
    ->gen_math_ops._sum()
        ->_op_def_lib._apply_op_helper #其中还涉及到tensorflow graph的构造
            -> op = g.create_op # 返回一个代表reduce_sum的op
        ->_result = _op.outputs[:] # reduce_sum运算的结果在op的output中

实际的运行在sess.run(op)中进行
sess.run 按graph结构，执行所有必要（当前op所依赖）的operation和tensor来得到当前op的结果

最终，调用_pywrap_tensorflow_internal.TF_Run(),这个函数也在_pywrap_tensorflow_internal.so中
```