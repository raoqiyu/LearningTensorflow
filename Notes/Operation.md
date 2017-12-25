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
 
```
reduction_ops.h
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
            -> op_info = self._ops.get(op_type_name, None)
                ...
            -> op_def = op_info.op_def
            -> op = g.create_op(...,op_def) # 返回一个代表reduce_sum的op
        ->_result = _op.outputs[:] # reduce_sum运算的结果在op的output中
   
```   
        
实际的运行在sess.run(op)中进行  
sess.run 按graph结构，执行所有必要（当前op所依赖）的operation和tensor来得到当前op的结果  
最终，调用_pywrap_tensorflow_internal.TF_Run(),这个函数也在_pywrap_tensorflow_internal.so中 


---
上面提到了与Operation相关的两个概念：kernel和ops  
- kernel 中定义了每个Operation具体的实现，然后REGISTER_KERNEL_BUILDER，将KernelRegistration注册到tensorflow中  
- ops 中定义了Operation的接口，然后REGISTER_OP将OpDefBuilderWrapper注册到tensorflow中

tensorflow的计算过程是先搭建一个graph，然后执行它。在搭建graph的过程中每个op会新建一个Node，而这个Node的建立就是基于ops中每个Operation注册的OpDefBuilderWrapper。执行过程中，每个Node会新建一个kernel，首先通过node查找到对应的op_def,然后该kernel的建立就是基于kernel注册的KernelRegistration。

---
#### Operation Interface

ops中实现了OpDefBuilderWrapper的注册与查找：
先看一下REGISTER_OP宏的定义

```C++
#tensorflow/core/framework/op.h

#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)
```
REGISTER_OP(name)，name就是具体的Operation的name返回一个OpDefBuilderWrapper，然后还可以继续设置其它属性，以SUM函数为例：

```C++
REGISTER_OP("Sum")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape)
    .Doc(R"doc(...));CreateOpKernel
        ->   const OpDef* op_def = nullptr;
```
具体注册流程如下：
```C++
REGISTER_OP(name)
    -> OpDefBuilderReceiver::OpDefBuilderReceiver
        -> OpRegistry::Global()->Register([wrapper](OpRegistrationData* op_reg_data)-> Status {
                return wrapper.builder().Finalize(op_reg_data);});
            -> 先用REGISTER_OP返回的OpDefBuilder生成一个OpRegistrationDataFactory的工厂op_data_factory
            -> 然后调用OpRegistry::Global()->Register(...)
                -> OpRegistry::Global() 返回都是OpRegistry* 的静态变量 global_op_registry
                -> global_op_registry->Register(op_data_factory)
                    -> 然后由op_data_factory生成一个OpRegistrationData类型的op_reg_data
                    -> gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name()
                        -> 插入到registry_中，关键字就是这个Op的name
                        -> registry_是一个map ： std::unordered_map<string, const OpRegistrationData*> registry_
```

然后在创建op时的查找流程，从python端开始：



```
op = g.create_op(..., op_def) # python端 tensorflow/python/framework/ops.py L1501
    -> node_def = _NodeDef(op_type, name, device=None, attrs=attrs) 
        -> 调用node_def_pb2生成一个NodeDef, 这个是由protocol buffer compiler生成的
        -> 与上面REGISTER_OP的NodeDefBuild用的是同一个定义
    
    有了NodeDef后就可以调用C_API新建一个C++端的Node
    -> self._c_op = self._create_c_op(graph, node_def, inputs, control_inputs)
        -> op_desc = c_api.TF_NewOperation(graph._c_graph,        # Op所属的graph                    
                                    compat.as_str(node_def.op),   # op_type, 就是operation实际的name
                                    compat.as_str(node_def.name)) # node name, (包括了variable scope)
            -> TF_NewOperation # C++端，tensorflow/c/c_api.cc
                -> TF_OperationDescription # tensorflow/c/c_api_internal.h
                    -> 保存该op相关的graph和NodeBuilder信息
                        -> NodeBuilder::NodeBuilder # tensorflow/core/graph/node_builder.cc 
                            -> 新建一个NodeDefBuilder
        -> 有了op_desc还要添加一些属性和输出等
        -> c_op = c_api.TF_FinishOperation(op_desc, status) # 得到最总的c_op
```
但是这个_c_op好像没什么用哎～～～～，只要偶NodeDef就行了，在Graph的笔记说明了在执行sess.run(..)之前需要将python的Graph转为GraphDef然后
更新到C++端，这个过程只用用到了了Graph中的NodeDef。    

然后在C++端创建kernel时，会通过NodeDef中的operation name来查找Opdef
```C++
CreateOpKernel # tensorflow/core/framework/op_kernel.cc L1059
    -> OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def); 
        -> 注册时用的是 OpRegistry::Global()->Register(...),这里用LookUpOpDef(...)
            -> OpRegistry::Global() 返回都是OpRegistry* 的静态变量 global_op_registry
            -> global_op_registry->LookUpOpDef(...)
                -> OpRegistry::LookUp(...) 在registry_找到匹配的OpRegistrationData
                -> 然后由该OpRegistrationData得到op_def
...
```


#### Operation Kernel
kernel包括各个Operation具体的实现以及相应kernel的注册： 先看一下REGISTER_KERNEL_BUILDER宏的定义


```C++
# tensorflow/core/framework/op_kernel.h


#define REGISTER_KERNEL_BUILDER(kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_builder, ...) \
  REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, __VA_ARGS__)

#define REGISTER_KERNEL_BUILDER_UNIQ(ctr, kernel_builder, ...)        \
  constexpr bool should_register_##ctr##__flag =                      \
      SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__);                        \
  static ::tensorflow::kernel_factory::OpKernelRegistrar              \
      registrar__body__##ctr##__object(                               \
          should_register_##ctr##__flag                               \
              ? ::tensorflow::register_kernel::kernel_builder.Build() \
              : nullptr,                                              \
          #__VA_ARGS__,                                               \
          [](::tensorflow::OpKernelConstruction* context)             \
              -> ::tensorflow::OpKernel* {                            \
            return new __VA_ARGS__(context);                          \
          });
```

REGISTER_KERNEL_BUILDER就是调用OpKernelRegistrar对Operation对应的KernelRegistration进行注册，需要三个参数:  
- Kernel_def(kernel的定义）
- kernel_class_name(kernel的名称)
- factory（用于生成该kernel的工厂）
，以SUM函数为例：

```C++
#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int32, Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int64, Eigen::internal::SumReducer<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
```
kernel_def的定义是通过Name来实现的，Name继承自KernelDefBuilder，设置了相应的属性后，调用Build()得到kernel_def

具体注册流程如下：

```C++
OpKernelRegistrar(const KernelDef* kernel_def, StringPiece kernel_class_name,
                    Factory factory) # tensorflow/core/framework/op_kernel.h
    -> OpKernelRegistrar::InitInternal 
        -> 生成key，由device_type，node_def和kernel label注册。
        -> GlobalKernelRegistryTyped()->insert(std::make_pair(key, KernelRegistration(*kernel_def, kernel_class_name, factory)));
            -> GlobalKernelRegistryTyped() 返回一个KernelRegistry*类型的静态变量 global_kernel_registry
            -> global_kernel_registry是一个map： typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry;
               然后将当前Op对应KernelRegistration插入其中
```
查找过程见Graph笔记