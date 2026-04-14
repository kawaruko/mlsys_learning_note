# 从mini-sglang的overlap scheduling来看cuda stream和cuda event


这几天在学习mini-sglang的时候，被一句很轻描淡写的话卡住了。在 [features.md](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/docs/features.md#L54-L59) 里，作者说 overlap scheduling 可以把 CPU scheduling overhead 和 GPU computation overlap 起来，从而提升吞吐。

这句话乍看一点都不复杂，可真想把它吃透时，问题马上就来了：GPU 到底靠什么知道哪些工作可以并行，哪些工作必须等？难道只是多 launch 几个 kernel，就已经足够表达任务级并行了吗？

问出来这个问题的时候，我意识到自己对更基础的执行模型其实并没有完全建立起来。于是这次我干脆退回到 NVIDIA 的 [CUDA C Programming Guide, Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) 这一章，重新理解 stream 和 event，再把这些概念带回 mini-sglang 的 overlap scheduling 实现。

这篇笔记只聚焦 stream 和 event，不展开 cooperative groups 和 atomic operations。

本文会按下面这条路线展开：

1. 先回答一个很具体的问题：stream 到底表达什么，event 到底解决什么。
2. 再用 CUDA 文档里经典的“三流”例子解释，为什么不能只是连续 launch 三个 kernel。
3. 接着回到 mini-sglang，看看 overlap scheduling 是怎样用两个 stream 加一个完成信号把依赖边界画出来的。
4. 最后顺手把这件事和 CUDA Graph 接起来，说明两者是上下游关系，不是谁替代谁。

照理，感谢 NVIDIA 官方文档、mini-sglang 这份相当清爽的代码，把serving 系统写得这么直白和清爽，便于我进一步理解sglang。

## Stream 表达的不是“更快”，而是“独立的工作队列”

如果只记一个结论，我觉得最值得记住的是下面这句：

> **stream 不是“让 kernel 自动并行”的开关，而是程序向 runtime 声明“这些工作彼此独立”的方式。**

NVIDIA 在文档的 Streams 一节里把 stream 定义为一串按顺序提交的 commands；同一个 stream 里的操作要按照提交顺序执行，而不同 stream 里的 commands 则可以乱序执行，或者并发执行，只是这种并发从来不是无条件保证的。[官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

这个定义看起来平平无奇，但它其实回答了 overlap scheduling 最核心的问题。因为一个 serving step 并不只有 kernel，它通常至少包含这样一条局部任务链：

1. host 侧准备 metadata，必要时把输入搬到 device。
2. device 侧运行 kernel 或一串 kernels。
3. 把结果异步搬回 host，供后续调度或回包逻辑消费。

注意到，这里有两种完全不同的“顺序关系”：

1. **块内顺序**：同一个 task 内部，H2D 必须先于 K，K 必须先于 D2H。
2. **块间独立**：不同 task 之间，原则上可以互不等待，只要硬件允许就可以 overlap。

stream 恰好就是用来同时表达这两件事的。你给每个 task 一条 stream，本质上是在说：“每条链内部有序，但链和链之间是独立的。”

### 同一条 stream 只能表达一条有序链

这里有一个很容易被忽略的点：**同一条 stream 非常擅长表达先后关系，但它天生不擅长表达独立性。**

如果所有操作都进同一条 stream，runtime 看到的就只是一条长队列：

1. `H2D_0`
2. `K_0`
3. `D2H_0`
4. `H2D_1`
5. `K_1`
6. `D2H_1`
7. `H2D_2`
8. `K_2`
9. `D2H_2`

这条队列当然是合法的，而且块内顺序也完全正确。但它还有一个副作用：**块间独立已经被你亲手抹掉了。** runtime 并不知道 `K_0` 和 `H2D_1` 之间其实没有必然依赖，于是它只能尊重你提交出来的全局顺序。

所以，问题的关键从来都不是“我 launch 了几个 kernel”，而是“我有没有把真正的依赖图表达出来”。

### 不同 stream 只是允许并发，不是保证并发

另一个经常被说快了的结论是：不同 stream 可以并发。

这句话如果只说前半句，其实是有风险的。更准确的说法应该是：

> **不同 stream 表达的是“允许并发”，不是“保证并发”。真正能不能 overlap，还取决于设备能力、提交顺序以及内存条件。**

NVIDIA 文档里明确提到，是否能并发 kernel，要看设备的 `concurrentKernels` 能力；是否能让 copy 和 compute overlap，要看 `asyncEngineCount`；而如果要让数据传输和 kernel 真正重叠，host 侧参与传输的内存还必须是 page-locked，也就是 pinned memory。[官方文档：Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution) [官方文档：Overlap of Data Transfer and Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlap-of-data-transfer-and-kernel-execution)

这也是为什么很多示例一上来就会用 `cudaHostAlloc`。它不是一个装饰性 API，而是在告诉 runtime：这段 host memory 可以参与真正的异步传输。

顺手再提一个很重要的小坑。CUDA 文档在 Implicit Synchronization 一节里还专门提醒：如果你在两个不同 stream 的操作中间，插入了 NULL stream 上的 CUDA 操作，那么这些原本可能并发的工作会被隐式同步掉，除非你使用的是 non-blocking streams。[官方文档：Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization)

所以你如果想处理好并发的话，记得注意这种小问题以免干扰并发度。

### Event 解决的不是“并行”，而是“完成边界”

如果说 stream 负责任务的并行和串行关系，那 event 负责的就是另一件事：

> **event 是一个完成标记，用来告诉其他 stream 或 host：某个边界之前的工作已经真的做完了。**

NVIDIA 文档对 event 的描述很准确：程序可以在某个时间点异步记录 event，然后查询它何时完成。一个 event 完成，意味着它之前的所有 task，或者某个 stream 中它之前的所有 commands，已经完成了。[官方文档：Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)

所以 event 常见有两类用法：

1. **跨 stream 建依赖**：例如 `cudaStreamWaitEvent()`，它的语义是“这条 stream 后续提交的工作，等这个 event 完成之后再执行”。[官方文档：Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#explicit-synchronization)
2. **做 timing 或完成确认**：例如在 kernel 前后记录 event，计算耗时；或者像后面 mini-sglang 那样，把 event 当成一次异步 D2H 完成后的安全消费边界。

从这个角度看，event 不是“另一个 stream”，也不是“更细粒度的并行工具”。它更像是给执行图打一个铆钉：这里之前的工作都完成了，你可以据此继续安排别的事情。

总的来说，stream 解决的是“独立工作怎么表达”，event 解决的是“完成边界怎么表达”。只有这两件事都讲清楚了，任务级并行才不会沦为一句空话。

## 三流例子里，真正并行的是三条任务链，而不是三个 kernel

CUDA 文档里有一个非常经典的例子：把长度为 30 的数组分成 3 块，每块 10 个元素，分别放进 3 条 stream 中，做异步 H2D、kernel 和异步 D2H。

下面这段代码是我按照文档思路重新整理过的版本。和你贴出来的逻辑一样，但我顺手把指针偏移写法改成了正常的 `+ i * chunk` 形式，避免 `sizeof(float)` 在指针运算里被重复乘进去。

<details>
<summary>三条 stream 的任务链示意代码</summary>

```cpp
int main() {
    constexpr int n_streams = 3;
    constexpr int chunk = 10;

    cudaStream_t streams[n_streams];
    float *A;
    float *d_A;

    for (int i = 0; i < n_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaHostAlloc(&A, n_streams * chunk * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_A, n_streams * chunk * sizeof(float));

    for (int i = 0; i < n_streams; ++i) {
        cudaMemcpyAsync(
            d_A + i * chunk,
            A + i * chunk,
            chunk * sizeof(float),
            cudaMemcpyHostToDevice,
            streams[i]);

        float_add<<<chunk, 1, 0, streams[i]>>>(d_A + i * chunk);

        cudaMemcpyAsync(
            A + i * chunk,
            d_A + i * chunk,
            chunk * sizeof(float),
            cudaMemcpyDeviceToHost,
            streams[i]);
    }
}
```

</details>

这段代码最值得看的地方，不是 `float_add<<<...>>>` 这行，而是它实际上向 runtime 提交了三条独立的任务链：

1. `stream[0]`：`H2D_0 -> K_0 -> D2H_0`
2. `stream[1]`：`H2D_1 -> K_1 -> D2H_1`
3. `stream[2]`：`H2D_2 -> K_2 -> D2H_2`

于是 runtime 终于看到了真正的结构：每条链内部有严格顺序，但不同链之间没有强制顺序。只要设备支持，`K_0` 就可能和 `H2D_1` overlap，`D2H_0` 也可能和别的链上的 compute overlap。NVIDIA 文档甚至专门指出，两个 stream 之间的实际 overlap 程度还会受到命令提交顺序影响；也就是说，**不是“开了 stream 就结束了”，还要看你是怎样把 commands 发进去的。** [官方文档：Overlapping Behavior](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlapping-behavior)

### 为什么不能只是同时 launch 三个 kernel？

有一个问题非常值得单独拿出来讲，因为它几乎就是 overlap scheduling 这篇文章的驱动问题：

> **为什么这个例子不可以只是连续 launch 三个 kernel，而一定要把它们放在不同的 stream 上？**

我觉得最核心的原因有三层。

第一层，如果这些操作仍然都放在同一条 stream 上，那么你 launch 的再多，本质上也还是在往同一条有序队列里塞命令。命令数变多了，但顺序关系没有被解除。runtime 看到的仍然是一条全局顺序链，而不是三条独立链。

第二层，你真正想表达的并不只是“有三个 kernel”。你想表达的是：每个 chunk 都有自己的一条 `H2D -> K -> D2H` 局部依赖链；与此同时，不同 chunk 之间又是独立的。单 stream 当然能表达“都按顺序做”，但它无法同时表达“块内有序”和“块间独立”。

第三层，copy 和 compute 能不能 overlap，关键不在于 kernel launch 的个数，而在于 runtime 有没有被清楚告知：`H2D_1` 并不需要等 `D2H_0`。只有把每个 chunk 放进独立的 stream，runtime 才有机会把这些本来独立的工作调度到一起执行。

换句话说，**stream 的价值不在“多开几条”，而在“把 DAG 画对”。**

下面这个对比表会更直观一些：

| 维度 | 单 stream 串行版 | 只改成连续 launch 三个 kernel | 三条 stream 的任务链版 |
|---|---|---|---|
| 能否表达块间独立 | 不能 | 不能，本质上仍是一条全局顺序队列 | 能 |
| 能否表达 H2D/K/D2H 的局部顺序 | 能，但只是“所有事全局排队” | 只能表达全局顺序，不能把每个 chunk 单独建模成独立链 | 能，每条链内部自然有序 |
| 是否可能出现 copy/compute overlap | 基本不能 | 基本不能，因为 copy 仍被同一条队列束缚 | 有机会，但取决于设备能力、提交顺序和 pinned memory |
| 是否依赖硬件能力 | 即使硬件支持也吃不到独立性 | 即使硬件支持也很难吃到 | 需要硬件支持才能真正 overlap，但至少把独立性表达出来了 |

把这个例子吃透以后，再看任何“任务级并行”的说法，脑子里都应该先出现一句话：**你到底是在多 launch 几个 kernel，还是在显式表达多条任务链？**

## 回到 mini-sglang：overlap scheduling 的依赖边界是怎么画出来的

现在我们回到最初那个问题。Mini-SGLang 在文档里说 overlap scheduling 会把 CPU scheduling overhead 和 GPU computation overlap 起来。[features.md](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/docs/features.md#L54-L59)

如果只看这句描述，很容易误以为它做的事情无非是：

1. 多发一点 GPU 工作。
2. 尽量别让 CPU 闲着。

真正读代码后会发现，它的实现比这两句话精确得多。它不是在模糊地“追求重叠”，而是在很明确地组织：

1. 哪些工作放在 scheduler 这条 stream 上做。
2. 哪些工作放在 engine 的 compute stream 上做。
3. compute stream 什么时候必须等 scheduler stream。
4. host 什么时候才能安全消费上一轮异步拷回来的结果。

### 第一步：Scheduler 额外创建一条 metadata stream，Engine 持有自己的 compute stream

先看 `Scheduler` 和 `Engine` 的初始化。

来自 [scheduler.py](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/scheduler/scheduler.py#L49-L55)：

```python
self.engine = Engine(config)
self.device = self.engine.device
self.stream = torch.cuda.Stream(device=self.device)
self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
torch.cuda.set_stream(self.stream)
```

来自 [engine.py](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/engine/engine.py#L35-L39)：

```python
self.device = torch.device(f"cuda:{config.tp_info.rank}")
torch.cuda.set_device(self.device)
torch.manual_seed(42)
self.stream = torch.cuda.Stream()
torch.cuda.set_stream(self.stream)
```

这一步其实已经把“谁和谁可以并行”表达出来了。

1. `Scheduler.stream` 负责 metadata 相关工作。
2. `Engine.stream` 负责真正的模型执行。

也就是说，mini-sglang 并没有把所有 CUDA 工作都塞进同一条默认执行线里，而是先在结构上承认：**metadata 准备和模型前向并不是同一类工作，它们值得被建模成两条不同的任务链。**

### 第二步：`wait_stream` 在设备侧画出“这里必须等”的边界

接着看 `Scheduler.overlap_loop()`。[scheduler.py](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/scheduler/scheduler.py#L83-L106)

```python
forward_input = self._schedule_next_batch()
ongoing_data = None
if forward_input is not None:
    with self.engine_stream_ctx:
        self.engine.stream.wait_stream(self.stream)
        ongoing_data = (forward_input, self._forward(forward_input))

self._process_last_data(last_data)
return ongoing_data
```

这里最关键的是 `self.engine.stream.wait_stream(self.stream)`。

从语义上，你完全可以把它理解成文档里 `cudaStreamWaitEvent()` 那类同步原语的一个高层封装：**compute stream 后续的工作，必须等 scheduler stream 里当前已经提交的工作完成之后，才能继续跑。**

这条边为什么必要？因为 `_schedule_next_batch()` 不是一个轻飘飘的 Python 函数调用，它背后会准备 batch、分配页表、构造 positions、准备 attention metadata 等等。如果这些 metadata 还没就绪，compute stream 就提前去读，那就不是 overlap scheduling，而是 data hazard 了。

所以这一步真正表达的是：

1. `Scheduler.stream` 和 `Engine.stream` 是独立的。
2. 但不是完全独立。
3. 在“metadata 准备完成”这个边界上，compute stream 必须等待。

有了这条边之后，GPU 才能既看到两条任务链的独立性，又知道哪里必须串起来。

### 第三步：异步 D2H 之后，要有一个“现在可以安全读了”的完成标记

前面那一步解决的是设备侧依赖。接下来要解决的，是 host 侧什么时候能安全消费上一轮结果。

先看 `Engine.forward_batch()`。[engine.py](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/engine/engine.py#L191-L206)

```python
next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)
return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

这里有两个细节连在一起看，味道就出来了。

第一，`next_tokens_gpu.to("cpu", non_blocking=True)` 发起的是异步 D2H。也就是说，`forward_batch()` 返回时，host 并不能假设 `next_tokens_cpu` 已经可安全读取。

第二，紧跟着记录一个 `copy_done_event`。这个 event 不是为了测速，而是为了给这次异步结果回传打一个完成边界：**当这个 event 完成时，我们才知道这次 copy 之前排在 compute stream 上的工作都做完了。**

然后再看 `Scheduler._process_last_data()`。[scheduler.py](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/scheduler/scheduler.py#L138-L167)

```python
batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
copy_done.synchronize()

for i, req in enumerate(batch.reqs):
    next_token = next_tokens_cpu[i]
    req.append_host(next_token.unsqueeze(0))
    ...
```

这段逻辑特别像我们前面讲的 event 定义。

`copy_done.synchronize()` 不是在说“所有 GPU 工作都停下来让我看看结果”，而是在说：**host 在真正读取 `next_tokens_cpu` 之前，先等这个完成标记落地。**

把这一段和 `overlap_loop()` 连起来看，整个时间关系就很清楚了：

1. 当前 batch 的 metadata 在 scheduler stream 上准备。
2. compute stream 等 metadata 准备完成，然后发起模型执行。
3. 当前 batch 的 GPU 结果异步拷回 CPU，并记录一个完成 event。
4. 在当前 batch 的 GPU 工作已经排队往前走之后，host 开始处理上一轮 `last_data`。
5. host 真正读取上一轮 `next_tokens_cpu` 之前，先等上一轮的 `copy_done_event` 完成。

这就是 overlap scheduling 最有意思的地方。它不是“CPU 和 GPU 永远同时忙”，而是：

> **把 GPU 正在执行当前 batch，和 host 正在收尾上一个 batch 这两件事，有意识地错开并重叠起来。**

换句话说，stream 在这里表达“metadata 准备”和“模型执行”是两条不同任务链；event 在这里表达“异步 D2H 的结果现在终于可以安全消费了”。

### 用一句话概括 mini-sglang 里的依赖图

如果把 mini-sglang 这一段实现压缩成一句话，我会写成：

> **scheduler stream 负责把下一批任务准备好，compute stream 负责把这一批任务算完，event 负责告诉 host：上一批结果现在真的可以读了。**

到这里再回头看开篇那个问题，答案就不再抽象了。mini-sglang 想要的从来不是“多 launch 几个 kernel”，而是“同时维护多条任务链，并把必须等待的边界明确画出来”。


## 后记

这次重新看 stream 和 event，我自己最大的收获反而不是记住了多少 API，而是把一个很朴素的判断标准重新建立起来了：

以后再看到“任务级并行”“overlap scheduling”“copy-compute overlap”这类说法，第一反应不该是“是不是多发了几个 kernel”，而应该是：

1. 这里的任务链到底是什么。
2. 哪些顺序是块内依赖，哪些是块间独立。
3. stream 有没有把这种独立性表达出来。
4. event 或同步原语有没有把完成边界表达出来。

一旦这四个问题问清楚了，再回头看 mini-sglang 的 overlap scheduling，我个人觉得它就不再神秘了。它只是非常老老实实地把任务图画对了而已。

## 参考

- [NVIDIA CUDA C Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- [NVIDIA CUDA C Programming Guide: Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [NVIDIA CUDA C Programming Guide: Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)
- [Mini-SGLang `features.md` at `20fcd7f`](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/docs/features.md#L54-L59)
- [Mini-SGLang `scheduler.py` at `20fcd7f`](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/scheduler/scheduler.py#L49-L55)
- [Mini-SGLang `scheduler.py` at `20fcd7f` (`overlap_loop` / `_process_last_data`)](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/scheduler/scheduler.py#L83-L167)
- [Mini-SGLang `engine.py` at `20fcd7f`](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/engine/engine.py#L35-L39)
- [Mini-SGLang `engine.py` at `20fcd7f` (`forward_batch`)](https://github.com/sgl-project/mini-sglang/blob/20fcd7f3c6a1898a61139fae1c97cc7a40bd8d66/python/minisgl/engine/engine.py#L191-L206)

<!-- /learn-write 自动检查报告
双轨检查：PASS
叙事检查：PASS
深度检查：understand-reproduce → understand-reproduce PASS
递进推导检查：PASS
交叉引用建议：当前仓库不包含已发布 CUDA Graph 系列正文文件，故未添加 repo 内相对路径交叉引用，避免产生失效链接。
-->
