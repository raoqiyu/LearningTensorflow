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
运行之前需要根据feed和fetch得到graph中，哪些部分是需要运行的
```C++
    -> DirectSession::GetOrCreateExecutors, 新建一个ExecutorsAndKeys参数，
        -> executor负责执行graph，keys,是指给每一个executor计算一个key，当每次运行时，通过查找key来判断是否
            已存在符合要求的executor，避免重复构造，key的构造是以feed和fetch作为关键字进行的
        -> DirectSession::CreateGraphs # 从当前最新的graph中构建ClientGraph
            -> execution_state->BuildGraph 
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


#### TODO 
Executor的执行过程