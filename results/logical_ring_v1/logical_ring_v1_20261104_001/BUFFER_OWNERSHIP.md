# G2a 缓冲区所有权与路由核对

本文件描述本轮新增 `src/gradient_comm_lab/logical_ring.py` 的实现边界；它不是网络传输或跨 stream 并行实现。实际检查结果以正式 correctness、message ledger 和 buffer lifetime 记录为准。G1 CUDA 源码及 a02 构建产物只读复用。

| 缓冲区 | 生产者 | 消费者 | 允许复用或释放时刻 |
|---|---|---|---|
| 调用者 input、threshold、LUT | 调用者/冻结夹具 provider | 入口及 codec | 本实现不原地修改这些输入；caller 不得在完成消费前主动覆盖 |
| prepared BF16 input | raw 入口先 cast，再 BF16 除 N；预除入口 clone | 本 rank 的全部 chunk 计算 | 当前 collective 结束后；返回的 prepared_inputs 另有快照 |
| padded local chunks | 每个 rank 独立分配、补零、复制 | RS/sink 的本地加数 | 当前 collective 结束后 |
| 单个 CudaCodecPlan 的 packet | 当前 encode | clone_packet 的三个 GPU clone | clone 已在同一 current stream 入队后，后续 encode 可复用 plan packet；stream 顺序保证 clone 先于覆盖 |
| 每次 encode 的 owned packet | clone_packet 独立克隆 payload、scale_codes、top_scales | RS 接收；sink 自解码；AG 全程转发；审计 | 全部存入 owned_packets，至少保留到 collective 结束，返回结果还持有它们；不主动覆盖 |
| CudaCodecPlan.output | 当前 decode | 立即执行的 GPU clone | clone 在同一 stream 入队后可被下一次 decode 覆盖 |
| decoded partial sum | decode output 的独立 clone | 当前 rank BF16 加法及审计 | 当前使用完成后；audit 保留独立值快照 |
| encode input/partial sum | 本地 chunk clone，必要时 BF16 add decoded | checked encode、审计 | 当前 encode 完成后；审计快照单独保留 |
| RS incoming / following 队列 | 前一轮发送、当前轮发送 | 当前轮接收、下一轮接收 | 两个不同 Python 队列；整轮所有发送结束才用 following 替换 incoming |
| AG outgoing / following 队列 | sink owned packets 或上一轮收到的同一包 | 本轮转发及接收解码 | 每轮全部发送后再解码接收，随后替换；只更换引用，不覆盖包、不再次编码 |
| 各 rank output chunks | sink self-decode 与 AG receiver decode | 最终裁剪、入口 dtype 转换 | 所有 chunk 填满后裁剪并 clone；raw 入口再转 FP32 |

`recordStream` 保护分配器生命周期，不阻止用户主动覆盖数据。本实现通过独立 packet/output clone 处理覆盖问题；两个措施用途不同。逻辑 hop barrier 是消息队列可见性的分界，不是每 hop 的全设备同步。当前实现按调用者的单一 CUDA current stream 顺序排入操作；没有把模拟 rank 映射成不同 CUDA stream。

## 原协议逐项对应

- RS hop 0：rank r 量化本地 chunk r，发送给 `(r+1) mod N`。
- RS hop h（1 至 N-2）：只读取前一轮 incoming，解码 chunk `(r-h) mod N`，以本地 chunk 的 clone 做 BF16 加法，再编码发送。
- Sink：rank r 消费前一轮 chunk `(r+1) mod N`，加本地值，再做第 N 次编码，并独立解码该 final packet 写入自身输出。
- AG hop h：发送 chunk `(r+1-h) mod N` 的原 final packet，接收 chunk `(r-h) mod N`；source_packet_id 固定为 sink 事件，发送原始包字节，不重新编码，不请求阈值。
- 对 N>1、D>0，每个 chunk 编码 N 次，总 N² 次；RS/AG 各 N(N-1) 次有向发送。Sink 编码不是额外发送轮。
- N=1 或 D=0 不加载/调用原生 codec、不访问 threshold provider、不生成消息；raw FP32 原样保留 dtype 和位模式。N>1、D>0 的全零流仍执行完整协议。

GPU 实现不调用 CPU oracle 的路由函数。host rank_order 只决定一轮中遍历 rank 的次序，事件键仍由 case、位宽、phase、rank、chunk、encoding_ordinal 决定；incoming 始终来自前一轮，不能读到当前轮新包。返回审计列表按稳定身份排序，不把 host 遍历顺序当协议身份。

## 字节及审计边界

每次 send 从实际 packet 三个张量的 `numel * element_size` 分别取得 payload、scale code、top scale 字节。发送队列承载 GPU 缓冲区引用，账本按有向逻辑发送逐次计数；这些字节不是测得的网络流量，也不是额外 CUDA copy 字节。审计 ID、哈希、路由字段不是已经实现的网络头，header_bytes=0。

编码输入、每次解码和部分和在 GPU 上 clone；外部 runner 才转 CPU 计算哈希和逐位差异。对冻结阈值/LUT 的 scalar finite 检查、构建身份核验和审计操作均属于验证路径，不构成 codec 热路径性能数据。本轮没有网络 AllReduce 延迟声明。

函数所有消息、计划、输出和事件均为调用内局部对象；没有跨 collective 消息缓存。发生拒绝或 codec 异常时函数直接抛出，不返回任何可被当作成功结果使用的部分聚合。外部 provider 自身的记录或副作用不属于函数内部状态；正式 provider 使用稳定事件键重建冻结阈值。

## 预冻结静态检查与证据定位

- `logs/actual_cuda_load_smoke.json`：主代理执行的实际 a02 加载及 N=4、D=1025、raw_fp32、b4/b8 smoke，检查 16 次编码、24 次发送及 rank 一致性。
- smoke 后仅新增两个入口约束：普通 dict 不作为 `.get(event)` provider；可选 LUT 必须逐位等于冻结 CPU 表。正式矩阵将绑定最终源码哈希，早期 smoke 保留原始身份。
- 完整在途 A/B 包复用、decode 输出复用、连续 collective、rank 顺序变换、输入/阈值/LUT 不变及非法输入拒绝结果由本轮 runner 另行记录；本文不预先宣称未执行的项目通过。
