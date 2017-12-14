##  Distributed TensorFlow
tensorflow中分布式训练实现，主要涉及以下几个目录
```
├── tensorflow
│   ├── distributed_runtime
│   │   ├── rpc
│   ├── public
│   ├── python
│   │   ├── client
│   │   ├── framework
│   │   ├── training
```

tensorflow进行分布式训练的一般步骤如下(详细代码见[github](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py))：

```
cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

...

if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                            config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

...

while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_step, global_step], feed_dict=train_feed)
      local_step += 1

```


tensorflow/distributed_runtime 定义了分布式版的**GrpcSession**,**GrpcServer**，以及Master，Woker，LocalMaster

![image](https://www.tensorflow.org/images/diag1.svg)

以[图](https://www.tensorflow.org/extend/architecture)中1个ps，1个worker为例，

- 两个job会各自启动一个GrpcServer，每个GrpcServer各有一个Master和Worker。

- /job:ps对应的GrpcServer会一直存在
- 有几个/job:worker就会有几个client，这里是1个client，client会建立GrpcSession，GrpcSession与/job:worker所在的**Master**进行通信
- client的GrpcSession::Run命令会通转为为Master.RunStep命令


---
上面的代码给出两种建立GrpcServer的方式，主要区别在于client与Master的连接方式

```
1. GrpServer和GrpcSession一起建立，FLAGS.existing_servers 为False

    - 此时GrpcServer与GrpcSession在同一个进程中（in-process），GrpcSession中包含一个LocaMaster；
    - GrpcServer的建立过程中会通过调用LocalMaster::Register对自己的Master进行注册,并且启动
       GrpcMasterService；
    - GrpcSession通过LocalMaster::Lookup(target)找到本client对应的Master，LocalMaster包含一个
      指向同一个进程内的Master
        target包含在GrpcSession创建时出入的SessOption参数；
    - tf.Session::run
        ->_pywrap_tensorflow_internal.TF_Run
            -> TF_Run_wrapper   # tensorflow/python/client/tf_session_helper.cc L144
                -> TF_Run       # tensorflow/c/c_api.cc L728
                    -> GrpcSession::Run
                        -> LocalMaster::RunStep
                            -> Master::RunStep


2. 预先创建GrpcServer，然后将GrpcServer的地址传给client，FLAGS.existing_servers为True

    - 此时GrpcServer与GrpcSession不在同一个进程中，GrpcSession中包含一个GrpcRemoteMaster；
    - GrpcServer的建立过程中同样会通过调用LocalMaster::Register对自己的Master进行注册，并且启动
       GrpcMasterService；
    - GrpcSession通过LocalMaster::Lookup(target)查找本client对应的Master，但是GrpcServer和
        GrpcSession不在一个进程中，LookUp失败，通过SessionOptio的target字段，建立GrpcChannel
        通过该channel建立GrpcRemoteMaster，GrpcRemoteMaster启动一个MasterService来连接
        GrpcServer的GrpcMasterService；
    - tf.Session::run
        -> _pywrap_tensorflow_internal.TF_Run
            -> TF_Run_wrapper   # tensorflow/python/client/tf_session_helper.cc L144
                -> TF_Run       # tensorflow/c/c_api.cc L728
                    -> GrpcSession::Run
                        -> GrpcRemoteMaster::RunStep
                            -> MasterService
                                -> Master::RunStep

```




---

### GrpcSession
GrpcSession 与 DirectSession相同，将自身的GrpcSessionFactory注册到SessionFactory，关键字为"GRPC_SESSION"

tensorflow/core/distributed_runtime/rpc/ 中包含了GrpcSession相关的实现

    - grpc_session.h,grpc_session.cc : GrpcSession, GrpcSessionFactory
        GrpcSession : 分布式版的Session, 运行与client端
        GrpcSessionFactory : GrpcSession的工厂类，继承自SessionFactory
            GrpcSessionFactory::AcceptsOptions, 根据SessionOptions.target 字段，判断是否是"GRPC_SESSION"
            GrpcSessionFactory::NewSession, 调用GrpcSession::Create创建一个GrpcSession
            GrpcSessionRegistrar: 将GrpcSessionFactory注册到SessionFactory中

    -

Python API 中定义了多种进行分布式训练的Session,例如

```
- tf.train.MoniteredTrainingSession               # tensorflow/python/training/monitored_session.py
        通过is_chief来区分chief worker和普通worker,对chief worker添加hook以方便对training过程
            进行monitor(如，CheckpointSaverHook、SummarySaverHook,StopAtStepHook),
            而普通worker则需要等待chief worker启动后在启动
        因此chief worker和普通worker的Session启动方式不同，
            主要是通过ChiefSessionCreator和WorkerSessionCreator来完成，
            而这两个Creator是通过tf.SessionManger的prepare_session和wait_for_session来完成

- tf.train.Supervisor.prepare_or_wait_for_session # tensorflow/python/training/supervisor.py
        与MoniteredTrainingSession类似，区分chief worker和普通worker
            也是通过调用tf.SessionManger的prepare_session和wait_for_session来完成
            但是，没有hook,　训练过程中的checkpoint和summary的save由用户写程序判断
-
```

可以看出Python API中分布式相关的Session最终都是由tf.train.SessionManager来完成，只不过是在上面进行封装，实现不同功能的Session
```
tf.train.SessionManager # tensorflow/python/training/session_manager
    - prepare_session　# 模型在当前时刻可用，则可以采用这个session,例如chief worker
        -> _restore_checkpoint
            -> 新建一个Session，与tf.Session的调用过程类似，只是在这里Session的target参数不为空,
                是这个client所要连接的Master的Grpc地址
            -> 从checkpoint中恢复session
        -> 进行初始化操作
    - wait_for_session # 模型不可用，采用这个session
        -> 新建一个Session，target参数是这个client所要连接的Master的Grpc地址
        -> 判断模型是否可用，（可以设置等待超时上限）
            -> run local init op,　成功后继续
            -> run ready op, 由参数传入，但一般默认用tf.get_collection(tf.GraphKeys.READY_OP)中的op
            -> 成功则返回，不成功则关闭当前session,然后继续判断

    - recover_session # 创建一个新session,若chenkpoint存在，则恢复，返回session
```
以wait_for_session为例，描述Python API与C++实现的对应关系
```
tf.train.SessionManager.wait_for_session() #tensorlow/python/training/session_manager.py
    -> session.Session()
        -> session.BaseSession()
            -> pywrap_tensorflow.TF_NewSessionOptions
            -> pywrap_tensorflow.TF_NewDeprecatedSession(opts, status) 在tensorflow.c.c_api.cc中，L326
                -> common_runtime.session::NewSession 在common_runtime.session.cc中，L6
                    -> SessionFactory::GetFactory
                    -> GetFactory通过遍历已经注册的SessionFactory，找到GrpcSessionFactory
                            （就是通过调用每个Factory的AcceptsOPtion()来判断使用那个Factory，
                              SessionOPtion.target参数为grpc地址，是GrpcSessionFactory）
                    -> GrpcSessionFactory::NewSession
                        -> GrpcSession::Create 返回一个新的GrpcSession
                -> 返回session
```


---
### GrpcServer
TODO