##  TensorFlow Device

```
├── tensorflow
│   ├── core
│   │   ├── common_runtime
│   │   ├── graph
│   │   ├── framework
│   ├── public
│   ├── python
│   │   ├── client
│   │   ├── framework
│   │   ├── training
```

### Python 端
Tensorflow 用于表示Device的类DeviceSpec，格式如下，有job,replica,task,device共4个字段

```
       /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
      or
       /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
```
device的用法主要有两个：
```
# 1.
with tf.device(device_name_or_function):
    ...

# 2.
with g.device(device_name_or_function):
    ...
```
第一种用法是在graph stack中找到当前的graph，然后调用该graph的device函数，与第二种类似。
device_name_or_function的用法参考[官方文档](https://www.tensorflow.org/versions/master/api_docs/python/tf/Graph#device)
```with g.device(...)```返回一个context manager，并且可以嵌套使用:

```
with g.device(...):
    ...
    with g.device(...):
        ...
```
嵌套使用时，当前层级的device_name_or_function会屏蔽上级设置
- 如果device_name_or_function是字符串，则新建一个DeviceSpec作为device_function,当对node分配device时，优先使用node_def.device设置
- 如果device_name_or_function为None或者Function，则使用当前device_name_or_function作为device_function

```
    with g.device(device_name_or_function):
        -> 按照上诉逻辑新建一个device_function
        -> 将新的device_function存入当前graph的device stack中，g._device_function_stack, 退出with时也会将其弹出
```
然后在创建Operation时，使用该devic_function为Operation分配device，整体流程见Operation笔记

```
op = g.create_op(..., op_def) # python端 tensorflow/python/framework/ops.py L1501
    -> ...
        ret = Operation(...) # 得到一个Operation
        ...
        self._apply_device_functions(ret)
            ->  def _apply_device_functions(self, op):
            ->      for device_function in reversed(self._device_function_stack):
                    if device_function is None:
                        break
                    op._set_device(device_function(op))  
                        -> self._node_def.device = _device_string(device)

```
python执行sess.run()时会将graph更新到C\++端，每个node的device信息也会更新到C++端
### C++端
C++端在运行时会为每个Node的Op建立一个OpKernel，建立OpKernel的过程就是查找已经注册好的kernel,，整体流程见Graph笔记

```
lib->CreateKernel(ndef, kernel);
    ->  FunctionLibraryRuntimeImpl::CreateKernel # tensorflow/core/common_runtime/function.cc
        -> CreateNonCachedKernel # tensorflow/core/common_runtime/executor.cc
            -> CreateOpKernel # tensorflow/core/framework/op_kernel.cc
                -> 根据node_def.op 查找op_def,验证op_def和node_def是否一致
                -> 找到对应的KernelRegistration， 
                    -> KernelRegistration包括：Kernel_def(kernel的定义）, kernel_class_name(kernel的名称)， factory（用于生成该kernel的工厂）
                    -> FindKernelRegistration # tensorflow/core/framework/op_kernel.cc
                        -> 根据kernel的device_type，node_def和label组成的key进行查找
                        -> 主要就是一个全局的静态变量static KernelRegistry* global_kernel_registry中进行查找
                            -> std::unordered_multimap<string, KernelRegistration> KernelRegistry，key到KernelRegistration的映射
                        -> 以上是查找已经注册好的OpKernel,具体的注册过程见Operation的笔记
                -> 找到对应的KernelRegistration后，构建一个OpKernelConstruction context，这是kernel执行所需的参数
                -> 然后调用KernelRegistration的factory新建一个该类型的kernel并返回
```
其中，查找时使用的key中包含了device_type信息,在注册时也会用到这个信息，注册过程见Operation笔记。





TODO
- device_mgr
- 
- 不同device的FunctionLibraryRuntime的实现，现在的理解：既然在注册和查找时考虑了不同的device_type，为啥还要对不同device注册不同的FunctionLibraryRuntime？





