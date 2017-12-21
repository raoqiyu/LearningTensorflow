##  TensorFlow Graph
tensorflow中Graph的实现，主要涉及以下几个目录
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


#### Python端
在使用Tensorflow的Python API时，Python API会维护一个graph stack(class _DefaultGraphStack)，用来保存现有的graph  
每次新建一个Tenor或者Op时，会从stack中查找当前的graph(就是栈顶的那个), python中经常会有如下对graph的操作:
```python
g=tf.Graph()
with g.as_default():
    .....
```
就是将g入栈，with 内部的操作都会从stack中找到g作为该操作所属的graph。  

当在python端运行sess.run()时，python端会调用```TF_ExtendGraph```将最新的graph更新到C++端的session中

#### C++端
C++端更新完graph后，根据python端传入的fetch和feed构建sub-graph（此外，还会有一个partition过程，如果graph是建在多个不同的device上时，将sub-graph再分成对应多个device的多个sub-graph），sub-graph中只包含必要运行的nodes（节约计算时间），每个sub-graph对应一个executor，随后由executor.RunAsync执行具体的操作。


---
#### Python端调用流程

python在导入tensorflow时会执行一系列的初始化操作，其中与graph相关的如下：

```python
import tensorflow as tf
    -> from tensorflow.python.framework.framework_lib import * # tensorflow/python/__init__.py
        -> from tensorflow.python.framework.ops import Graph   # tensorflow/python/framework_lib.py
            -> _default_graph_stack = _DefaultGraphStack()     # tensorflow/python/ops.py   
                -> 这个graph stack是继承自_DefaultStack（包括一些栈的操作方法）
```
在创建tensorflow相关的变量时也会涉及到grap的操作
```python
tf.placeholder(...)
    -> array_ops.placeholder                                   # tensorflow/python/ops/array_ops.py
        -> gen_array_ops._placeholder                          # tensorflow/python/ops/gen_array_ops.py
            # 不同的Op最终都会调用下面的这个_apply_op_helper操作，因此graph的操作类似
            -> _op_def_lib._apply_op_helper                    # tensorflow/python/framework/op_def_library.py
                -> g = ops._get_graph_from_inputs(...)
                    -> #这个函数会从传入的输入参数中推断当前的graph，并且验证输入与当前op是否在同一graph中
                    -># 若输入没有graph(如python int类型参数)，则调用get_default_graph()
                        -> _default_graph_stack.get_default()
                            -> #这会从 graph stack取出栈顶的graph，若不存在则
                            -> #调用_DefaultGraphStack._GetGlobalDefaultGraph创建一个_global_default_graph并返回
                -> with g.as_default(), ops.name_scope(name) as scope: # 将上诉过程得到的g，入栈
                       op = g.create_op(...) # 这里进行都会与graph g进行关联（例如，device 分配，node的创建等）
                -> #退出with后,g也会从_default_graph_stack中弹出，
                -> #返回op
```                
当继续创建其它tensorflow变量时，因为_global_default_graph的存在，  
所有的变量都会在一个graph中，除非_default_graph_stack中存在其它graph，例如用户主动创建graph，并入栈，
```python
g=tf.Graph()，  
with g.as_default()):
```                         
在python端调用sess.run()时，涉及到将当前最新的graph更新到C++端的session
```python
sess = tf.Session()  
    -> self._graph = ops.get_default_graph(), #在创建session时，若参数中没有传入graph，
    -> #则从_default_graph_stack中获取当前所处环境的graph
sess.run(fetches, feed_dict, options, run_metadata)
    -> Session._extend_graph()  # 获取当前graph的def，然后传给C++端的session
        -> graph_def, self._current_version = self._graph._as_graph_def(...)
        -> _pywrap_tensorflow_internal.TF_ExtendGraph(self._session, graph_def.SerializeToString(), status)
    -> _pywrap_tensorflow_internal.TF_Run 
        -> TF_Run_wrapper   # tensorflow/python/client/tf_session_helper.cc L144， swig实现的pywraper
            -> TF_Run       # tensorflow/c/c_api.cc L728
                -> DirectSession::Run
```


---

#### C++端实现流程
C++中Graph的定义在```tensorflow/core/graph/graph.h ```中
总的来说Graph是通过有向图来实现的，图中有Node和Egde， Node和Edge都有表示自己的id  
Node中定义着Operation，每个Node可以有多个输入和输出  
- in_edges_ 连接本Node的输入的Edge集合
- out_edges_ 连接本Node的输出的Edge集合
- graph_    本Node所属的Graph
- NodeProperties: OpDef, Nodedef, input_types, output_types

Edge用来连接不同Node的输入和输出
- src_表示输入节点，src_output_表示本Edge连接的是输入节点的第几个输出
- dst_表示输入节点，dst_input_表示本Edge连接的是输出节点的第几个输入    

Graph中有两个特殊的Node：
- source\_node： node id 为0，构建sub-graph时，source_node连向所有没有输入的node
- sink\_node  ： node id 为1,构建sub-graph时，所有没有输出的node都连向这个sink_node


DirectSession有一个GraphExecutionState类型的指针参数**execution_state_**，负责生成一个可执行的ClientGraph，它是session所拥有的grap的一个sub-graph 

当python端调用```TF_ExtendGraph```后，C++端对session的graph进行更新

```C++
TF_ExtendGraph   # tensorflow/c/c_api.cc L348, tensorflow的实现逻辑中TF_*的操作一般是通过swig对c_api进行包装
    -> session->Extend(g)  # 以DirectSession为例， tensorflow/core/common_runtime/direct_session.cc L381
        -> 获取graph锁,调用GraphExecutionState::Extend,然后释放锁
            -> GraphExecutionState::Extend(extension_def,out） #tensorflow/core/common_runtime/graph_execution_state.cc
                -> 新建一个GraphDef gdef，从该函数代码注释看主要分7步
                -> 1. Copy the function library. 
                    function library的作用有待研究，将function library拷贝到gdef
                    
                -> 2. Build an index of the new node names. 
                    就是按顺序从extension_def中依次每个node的name
                    
                -> 3. Add the non-duplicates from the old graph to the new graph.
                      Return an error if the same node name appears in both the
                      old graph and the extension.
                    将old graph中每个node复制到gde中，若发现old graph中的node和extension_def中的相同，
                    则报错（就是查找old graph的node.name()是否在第2步构建的index中）
                    
                -> 4. Merge the versions field.
                    将extension_def.node()与gdef的node进行merge(具体怎么merge待研究)
                    然后检查gdef的version与extension_def的version是否相同，不同则返回作物状态
                    
                -> 5. Validate that the final graphdef is valid.
                    调用graph::ValidateGraphDef(gdef, *flib_def_)，检查gdef是否可用
                    
                -> 6. Add the extension.
                    增加其它属性，如device_set,sessison_option等，然后新建一个new_execution_state并返回

```
有了最新的graph后，就可以根据python端sess.run传来的参数进行相应的操作，这里只看graph相关的过程。  
如上所诉，session中有一个GraphExecutionState类型的参数，其中保存graph相关内容。此外还有一个ExecutorsAndKeys(定义在direct_session.h中)
类型的参数用来保存执行graph时所需的内容。

```C++
DirectSession::Run
```
运行之前需要根据feed和fetch得到graph中，哪些部分（sub-graph)是需要运行(executor),sub-graph的构建过程如下：
```C++
    -> DirectSession::GetOrCreateExecutors, 新建一个ExecutorsAndKeys参数，
        -> executor负责执行graph，keys,是指给每一个executor计算一个key，当每次运行时，通过查找key来判断是否
            已存在符合要求的executor，避免重复构造，key的构造是以feed和fetch作为关键字进行的
        -> DirectSession::CreateGraphs # 从当前最新的graph中构建ClientGraph
            -> execution_state->BuildGraph  # tensorflow/core/common_runtime/graph_execution_state.cc
                从execution_state->graph_中截取必要的部分，构成sub-graph，然后作为Client返回给executor
                首先将完整的graph复制一份过来，其中还涉及到优化的操作GraphExecutionState::OptimizeGraph
                然后对复制的graph进行Rewrite得到一份可执行的graph
                -> subgraph::RewriteGraphForExecution 
                    -> FeedInputs
                        对Feed中的每个参数，新建一个Node，关键字为_Arg，然后取代原有的Node
                        替换原则是：对原有Node的output连接的所有其它节点，新Node同样连接到这些节点，但是原Node的
                                    input连接的节点都不再连接（添加连接时还要分数据流Edge和控制流ControlEdge）
                        这样新Node只有输入的Edge，没有输出的Edge
                        然后删除原有节点
                    -> FetchOutputs
                        对Fetch中的每个参数，新建一个Node，关键字为_Retval
                        然后原Node的输出连接到新Node的输入，并且在新Node与graph的sink_node之间添加一条ControlEdge
                        然后将新Node作为Fetch参数对应的out_fetch_nodes
                    -> PruneForTargets
                        修剪graph，删除无用的Node和Edge
                        首先将fetch_nodes和target_nodes合并到targets中
                        -> PruneForReverseReachability # tensorflow/core/graph/algorithm.cc
                            以targets中的每个Node作为其实节点做breadth-first search
                            搜索完成后，删除所有没有遍历到的Node
                        -> FixupSourceAndSinkEdges # tensorflow/core/graph/algorithm.cc
                            对所有没有输入的Node，添加一条从source_node连接到它们的ControlEdge
                            对所有没有输出的Node，添加一条从它们连接到sink_node的ControlEdge
                        -> 返回修剪后的graph
                -> 将其它信息（session_options, flib_def, device_set等）连同graph，添加到ClientGraph中
```
通过上诉步骤得到用于执行的ClientGraph后，还需考虑graph中node在不同device上的情况，

```C++
            -> Partition(popts, &client_graph->graph, &partitions) # tensorflow/core/common_runtime/direct_session.cc L 1374
                Partition 函数在 tensorflow/core/graph/graph_partition.h
                根据每个node的device_name分配到不同的graph中
                popts.node_to_loc = [](const Node* node) {
                    return node->assigned_device_name();
                };
                .....
                std::unordered_map<string, GraphDef> partitions;
                ...
                # op_nodes()返回的nodes是按照node id顺序排列的
                for (const Node* dst : g->op_nodes()) {
                    # 遍历graph中所有node，找到其所属的partition，
                    dstp = opts.node_to_loc(dst);
                    GraphDef* dst_graph = &(*partitions)[dstp];
                    # 添加到所属partition
                    NodeDef* dst_def = dst_graph->add_node();
                    *dst_def = dst->def();
                    ...
                    通过对dst的in_edges进行遍历，遍历所有的上游Node，
                        判断上游Node与dst是否在同一个partition中
                            如果在直接，且不需要end/recv机制，则添加连接
                            否则，则添加一个send/recv pair transferring，
                                先在上游Node src所在的graph添加一个dummy node，src连向dummy node
                                然后对dummy node的输出上添加一个send node
                                然后在Node dst所在graph体检一个recv node， recv node与send node相连
                                然后recv node的输出连向Node dst
                            其中，send/recv node的搭建由NodeDefBuilder完成 # tensorlow/core/framework/node_def_build.h
                                node中的op可以是以下集中，_HostSend, _Send, _HostRecv, _Recv, 
                    
                    其中还需要考虑ControlEdge,还有graph的其它信息，如FunctionLibraryDefinition， versions， 

```
Partition操作完成后，得到多个GraphDef，然后将其转为graph,然后写到相应的device上（device部分还没有看，具体逻辑不清楚）
```C++
            -> for (const auto& partition : partitions) {
                   std::unique_ptr<Graph> device_graph(
                                new Graph(client_graph->flib_def.get()));
                   ...
                   ConvertGraphDefToGraph(device_opts, partition.second,
                                                  device_graph.get())
                                                  
                ...
                
                Device* d;
                s = device_mgr_->LookupDevice(partition_name, &d);
            
            -> 然后返回所有的partition
                
```
得到所需的Graph后，分配executor，这样ExecutorsAndKeys就创建完成了

```C++
        -> for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
                const string& partition_name = iter->first;
                std::unique_ptr<Graph>& partition_graph = iter->second;
               ...
                item->graph = partition_graph.get();
                item->executor = nullptr;
                Executor* executor;
                TF_RETURN_IF_ERROR(
                    NewLocalExecutor(params, partition_graph.release(), &executor));
            }
```


---

```GetOrCreateExecutors``` 操作完成后，每个sub-graph都有自己的executor，接下来就由executor执行各自负责的graph，executor的大致流程为：  

- executor的创建过程       
    -   将graph转为bytes格式的GraphView，其中包含了NodeItem， NodeItem比Node多了一些kernel的信息
    -   根据graph中的controlflow生成frame，每个node都计算pending counts（pending count就是node要等待多少个输入才会进入ready状态  
        这里的frame的作用还没搞清楚，感觉像是循环用到概念，每次迭代都是一个frame，frame保存每一次迭代的状态
    -   会对自己的graph中每个node建立一个Opkernel（每个operation对应的kernel在tensorflow/core/kernel/中进行注册）
        kernel创建结束后，还会对当前NodeItem添加一些属性：Merge，Enter, Exit, Sink等状态属性（TODO 属性的具体含义待研究）
    
- executor->RunAsync过程
    - 将没有输入Egde的Node加入到root_nodes集合中（如source_node)
    - 对root_nodes中的每个node新建一个TaggedNode（TaggedNode还有node，frame，iter等信息），并加入ready队列
    - 起一个线程调用Process依次对ready队列中TaggedNode中的node进行运算，每个node运算完成后会对其输出进行处理，  
      并且将当前node的输出边连接的nodes中pending count为0的node加入新的ready队列
    - 对新的ready队列进行处理


上诉执行流程涉及到几个概念

为node建立Opkernel的过程，主要就是调用creat_kernel函数，根据node的定义NodeDef来创建kernel,流程如下：

```C++
ek->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_.get(), options_.env, graph_def_version, ek->flib_def.get(),
      optimizer_opts));   # 根据device_mgr_的相关信息对每种Device都新建一个NewFunctionLibraryRuntime
auto lib = ek->proc_flr->GetFLR(partition_name); # partition_name就是上面所得partition过程的得到的，其实就是device_name，
                                                 # 上一语句建立，这里查找并使用
...
LocalExecutorParams params;
...
params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // Caches the kernel only if the node is stateful.
      LOG(INFO) << "FunctionLibraryRuntimeImpl::CreateKernel . custom_kernel_creator out\n";
      if (!lib->IsStateful(ndef.op())) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
```
其中lib是FunctionLibraryRuntime，根据device_name得到相应的FunctionLibraryRuntime，这里好像只有tensorflow/core/common_runtime/function.cc中的FunctionLibraryRuntimeImpl实现，具体创建kernel的过程如下：

```C++
lib->CreateKernel(ndef, kernel);
    ->  FunctionLibraryRuntimeImpl::CreateKernel # tensorflow/core/common_runtime/function.cc
        -> CreateNonCachedKernel # tensorflow/core/common_runtime/executor.cc
            -> CreateOpKernel # tensorflow/core/framework/op_kernel.cc
                -> 根据node_def.op 查找op_def,
                -> 找到op_def找到对应的KernelRegistration， 
                    -> KernelRegistration包括：Kernel_def(kernel的定义）, kernel_class_name(kernel的名称)， factory（用于生成该kernel的工厂）
                    -> FindKernelRegistration # tensorflow/core/framework/op_kernel.cc
                        -> 根据kernel的device_type，node_def和label组成的key进行查找
                        -> 主要就是一个全局的静态变量static KernelRegistry* global_kernel_registry中进行查找
                            -> std::unordered_multimap<string, KernelRegistration> KernelRegistry，key到KernelRegistration的映射
                        -> 以上是查找已经注册好的OpKernel,具体的注册过程见Operation的笔记
                -> 找到对应的KernelRegistration后，构建一个OpKernelConstruction context，这是kernel执行所需的参数
                -> 然后调用KernelRegistration的factory新建一个该类型的kernel并返回
```
---
接下来便是executor具体的执行流程，这里涉及一个较为重要的概念: ```FunctionCallFrame```
```
// FunctionCallFrame: 就是对feed和fetch的数据进行操作
// Represents a function call frame. I.e., the data structure used to
// pass arguments to a function and retrieve its results.
// Runtime must arrange accesses to one FunctionCallFrame s.t.
//   1. SetArgs() happens before any GetArg();
//   2. GetRetvals happens after all SetRetval();
FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
...
const Status s = call_frame.SetArgs(feed_args);
...
args.call_frame = &call_frame;
...
for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }
std::vector<Tensor> sorted_outputs;
const Status s = call_frame.ConsumeRetvals(&sorted_outputs);
```
在executor执行之前，设置feed数据，执行结束之后，在获取fetch数据。在上面提到的sub-graph的构建过程的中FeedInputs和FetchOutputs操作中，新建两种分别用于处理feed和fetch的Node: _Arg 和 _Retval,这两种Node对应的Operation定义在tensorflow/core/kernels/function_ops.cc中

```
class ArgOp : public OpKernel {
  ...
  void Compute(OpKernelContext* ctx) override {
    auto frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
    Tensor val;
    OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
    ...
    ctx->set_output(0, val);
  }
    ...
};
```
上面的代码片段就是ArgOp的主要执行逻辑，调用call_frame.GetArg来时获取用户feed的数据，因此在Run之前要先SetArg

```C++
class RetvalOp : public OpKernel {
  ...
  void Compute(OpKernelContext* ctx) override {
    const Tensor& val = ctx->input(0);
    auto frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
    OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
  }
 ...
};
```
RetvalOp主要就是调用call_frame.SetRetval来时设置用户fetch的数据，然后可以通过GetRetval得到想要的数据  
两种Op都是通过index_进行数据索引，是因为在executor的的建立过程中feed和fetch都经过排序，然后才建立sub-graph的，然后在FeedInputs和FetchOutputs中建立Node时，会将排序后的顺序索引作为属性传递给Node

---

```C++
item.executor->RunAsync(args, barrier->Get());
    -> ExecutorImpl::RunAsync(const Args& args, DoneCallback done) 
        -> 新建一个ExecutorState(args, this)
        -> 然后调用它的RunAsync(std::move(done))
        -> ExecutorState::RunAsync(done) 
            -> 填充device (TODO 为啥填充，还没研究)
            -> 对root_nodes中的node建立Taggednode,然后加入ready队列
            -> ScheduleReady(ready, nullptr); # 处理ready队列
                -> runner_([=]() { Process(tagged_node, scheduled_usec); }); #其中好多分支，但是ScheduleReady的第二参数为nullptr,因此走这里
                    -> ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec)
                        -> 首先就是新建一个OpKernelContext::Params params，其包含诸如device，session_state, call_frame, inputs等信息
```
获取输入，在Op的操作中（如上面两个Op），输入和输出是通过ctx->input和ctx->output来传递的，                          因此在run之前要对params的input字段进行填充，这个主要由GetInputTensors和PrepareInputs两个函数来完成
```C++                  
                        -> 获取输入
                            -> GetInputTensors
                                -> 获取当前node所在frame的IterationState的input_tensors
                            -> PrepareInputs
                                -> 根据input_tensors的起始地址，将input_tensors中的val填入params.inputs
                                -> 如果得到的输入不够，则可能是已经有其它支路运行完成（MaybeMarkCompleted）
                                    然后运行NodeDonw。之后处理下一个TaggedNode, 而不进行实际的运算
                            (TODO 这部分待研究)
```
进行实际的kernel计算
```C++
                        -> 然后由params新建一个OpKernelContext， 然后调用device->compute
                                OpKernelContext ctx(&params, item.num_outputs);
                                nodestats::SetOpStart(stats);
                                device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
                                    -> device的compute方法就是调用op_kernel的compute方法并将ctx传给它
                                       op_kernel->Compute(context)
                            计算完成后，如果op_kernel有输出，会放在ctx->output
```
然后处理outputs，由ProcessOutputs和PropagateOutputs负责
```C++
                        -> ExecutorState::ProcessOutputs,还要做一些类型检查，然后将ctx->output中的数据拿出来
                        -> PropagateOutputs负责更新所有的下游Node:
                            ExecutorState::PropagateOutputs
                                -> 根据当前node的状态，选择不同的处理方式，主要是FrameState的更新，是还处在当前iter还是下一iter
                                -> 然后FrameState::ActivateNodes来激活下一个FrameState中的node
                                    -> ExecutorState::FrameState::ActivateNodes
                                        -> 对当前Node的输出边连接的所有下游节点：
                                            如果该下游节点：
                                                如果是Control Edge则减少下游节点的pending counts次数2次
                                                如果是普通的Edge，则还要区分当前的output中时候含有该下游节点需要的输出
                                                    如果含有，则减少下游节点的pending counts次数1次，并且判断该下游节点
                                                            是否需要这个输出
                                                    如果不含有，则dead enter的判断，这部分待研究
                                            否则：
                                                TODO 待研究
                                                其它的处理方式，总之都是处理pending counts还有是否需要这个输出
                                            
                                            -> 如果需要该输出,则将该输出填充到FrameState的IterationState的input_tensors
                                                (这与上面PrepareInputs的操作是对应的，这里处理每个node时将输出填充到下一次运算的Frame中，
                                                然后下一次运算时，PrepareInputs从中提取数据)
                                                
                                            -> 如果pending counts为0，则将该下游节点加入新的ready队列，并且将下个FrameState的IterationState
                                               的outstanding_ops数加1
                                -> 对当前ready队列中的node进行Propagate时都要判断当前Frame是否运行结束了，如果结束就要对Frame进行扫尾工作
```
node计算结束后，进行扫尾工作，有NodeDone负责，
```C++
                            ExecutorState::NodeDone
                                -> 更新当前FrameState的IterationState的num_outstanding_ops_
                                    若果num_outstanding_ops_为0，则当前Frame计算完成，
                                -> 如果当前node计算正常，返回的ctx->status().ok()为True，
                                    对PropagateOutputs得到的ready队列调用ScheduleReady进行操作
                        -> 如果Frame计算完成，调用Finish(),结束
```
上诉过程，每个executor都执行一遍，然后开始等待执行结束，如下：
```C++
  for (const auto& item : executors_and_keys->items) {
    item.executor->RunAsync(args, barrier->Get());
  }
  WaitForNotification(&run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);
```
结束后，需要fetch的数据就会由```_RetVal```节点的RetvalOp写入到call_frame中，
```
    std::vector<Tensor> sorted_outputs;
    const Status s = call_frame.ConsumeRetvals(&sorted_outputs);
```
通过call_frame.ConsumeRetvals得到输出并返回


#### TODO 
- 在BuildGraph中涉及到GraphExecutionState::OptimizeGraph操作  
- ConstOp的数据存储方式  
- Executor初始化过程中的BuildControlFlowInfo  
- FrameState的作用  
- PrepareInputs的具体逻辑  
- NodeItem一些属性：Merge，Enter, Exit, Sink等状态属性的作用