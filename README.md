## Todo

训练 model 阶段用 pytorch + nn + flash-attn 调包版，推理阶段逐步替换为自己的实现

- [x] Pytorch + nn 版 Qwen3.8 27b 推理，单 token 推理
- [x] Pytorch + nn 版 Qwen3.8 27b 推理，单请求 Prefill + Decode：Attn KV Cache + Linear Attn Cache
- [ ] Pytorch + nn 版 Qwen3.8 27b 推理，采样与参数控制：SamplingParams + Sampler
- [ ] Pytorch + nn 版 Qwen3.8 27b 推理，分页显存管理：BlockManager + Paged KV Cache
- [ ] Pytorch + nn 版 Qwen3.8 27b 推理，连续批处理调度：Sequence + Scheduler
- [ ] Pytorch + nn 版 Qwen3.8 27b 推理，批处理执行与图加速：ModelRunner + Context + flash-attn + CUDA Graph
- [ ] Pytorch + nn 版 Qwen3.8 27b 推理，引擎外层封装：LLMEngine + Tokenizer
- [ ] 造数据
- [ ] Benchmark
- [ ] 复现 DSpark Draft Model
- [ ] 设计 Copy Locator Model 架构和 Router 架构
- [ ] 训练以上 3 个 Model 模型
- [ ] 将基础推理框架替换为个人算子库
- [ ] 实现自定义本项目特化的算子

