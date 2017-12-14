## Tensorflow Session
tensorflow中Session的实现，主要涉及以下几个目录
```
├── tensorflow
│   ├── common_runtime
│   ├── public
│   ├── python
│   │   ├── client
```

#### tensorflow/public
```
    - session.h 定义了Session类
    - session_options.h 定义了结构体SessionOptions，用于保存Session information
                        其中SessionOption.target表明了运行Session的engine，
                        后续各种不同Session的实现就是通过这个参数进行区分
```
#### tensorflow/common_runtime 定义了单机版的**DirectSession**，以及其它运行时环境

```
    - session_factory.h,session_factory.cc:  Session的工厂类
            std::unordered_map<string, SessionFactory*> SessionFactories
            SessionFactory::Register() : 将不同实现的SessionFactory添加到SessionFactories中
            SessionFactory::GetFactory(SessionOptions) : 遍历SessionFactories，
                                        调用每个SessionFactories的AcceptsOptions,根据参数
                                        SessionOptions.target中的参数判断使用那个工厂
            SessionFactory::AcceptsOptions(SessionOptions) : 由各个具体实现的工程类实现


    - direct_session.h,direct_session.cc:  DirectSession，DirectSessionFactory
            DirectSession       : 单机版的Session，in-process
            DirectSessionFactory: DirectSession的工厂类，继承自SessionFactory
                DirectSessionFactory::AcceptsOptions(SessionOptions): SessionOptions.target为空，
                                                                        则适用当前Session
                DirectSessionFactory::NewSession(SessionOptions) : 创建一个新的DirectSession

            DirectSessionRegistrar: 将DirectSessionFactory注册到SessionFactory中


```


例子：
tf.Session()的调用过程,单机版Session的创建，其中包含的graph和device等操作，忽略，主要看创建Session的总体流程


```
  tf.Session() 在tensorlow.python.client.session.py中 ,target参数为空
  session.Session()
    -> session.BaseSession()
        -> pywrap_tensorflow.TF_NewSessionOptions
        -> pywrap_tensorflow.TF_NewDeprecatedSession(opts, status) 在tensorflow.c.c_api.cc中，L326
            -> common_runtime.session::NewSession 在common_runtime.session.cc中，L6
                -> SessionFactory::GetFactory
                    -> GetFactory通过便利已经注册的SessionFactory，找到DirectSessionFactory
                            （就是通过调用每个Factory的AcceptsOPtion()来判断使用那个Factory，
                              target参数为空，是DirectSessionFactory）
                -> factory->NewSession 返回一个新的DirectSession
            -> 返回session

   - tf.Session::run
        ->_pywrap_tensorflow_internal.TF_Run
            -> TF_Run_wrapper   # tensorflow/python/client/tf_session_helper.cc L144
                -> TF_Run       # tensorflow/c/c_api.cc L728
                    -> DirectSession::Run

```
