# 在 Orin 上为 lossless 9 Hz 优化 Pi0.5：三个被推翻的"常识"

> 背景：Pi0.5 是 Physical Intelligence 开源的 VLA 控制模型，由 **PaliGemma-3B**（SigLIP-So400m 视觉 + Gemma-2B encoder）加上自己的 **300M 扩散 decoder** 组成。FlashRT 是它的实时推理引擎。本文跑在 Jetson AGX Orin 64GB（SM87，16 SMs，LPDDR5X 204 GB/s，无 native FP8）上。

起点 128 ms / **7.81 Hz**，目标**严格 bit-equivalent**（cosine = 1.000、每帧 byte-identical）9 Hz。最终撞到 **8.04 Hz** lossless 上限——9 Hz 数学上过不去。这篇不写优化技巧，只挑三个让我反复在错误结论上原地打转的常识。

## 1. "ncu 报 92% util 就饱和"——错。

文档原本写 "encoder gate_up 92% utilization，没软件空间"。这话是单 kernel 瞬时 ncu 数据。

我用 roofline + per-shape microbench 重新做：

| Shape | 利用率 |
|---|---:|
| enc_qkv (522,2560,2048) | **23%** |
| enc_o (522,2048,2048) | 48% |
| enc_gate/up (522,8192,2048) | **88%** |
| enc_down (522,2048,16384) | 50% |

`gate/up` 确实饱和，但 `qkv` 只有 23%，被 128×128 tile 在 N=2560 上撞出 6.25 wave 的 partial 给坑了。

**下次看到 "X% util" 先问：哪个 kernel？哪个 shape？瞬时还是平均？**

## 2. "改 GEMM tile shape 必破 bit-equality"——错。

我加了一个 64×128 tile 给 qkv 这种 wave-不齐 shape。第一次测 cosine 0.96-0.99，得出"tile change 必破 bit-equality"的结论，然后陷在 strict-precision 各种 fusion 里几小时。

后来重新干净测——是 baseline.npz 没包含中间步骤的 fusion 改动，**不是 tile change 本身的问题**。

真相：CUTLASS INT8 GEMM 的 INT32 累加是结合的（K=2048×127² ≈ 33M ≪ 2³¹，不溢出），EVT fp32 dequant epilogue 的乘法顺序与 tile shape **无关**。所以**每个 GEMM shape 都可以独立选最优 tile，不破 lossless**。

最终落地的 per-shape dispatch 节省 2.1 ms，6/6 帧 maxabs=0：

```cpp
if (M <= 64) return ...t64x128::run(...);   // decoder M=10
if (N > 2048 && N <= 4096) return ...t64x128::run(...);  // qkv-shape
return ...::run(...);  // default 128×128
```

**INT 累加结合性 + EVT 形状无关 = tile shape 是免费的优化空间**。

## 3. "microbench 2-3× 加速 → pipeline 也快 2-3×"——错。

我写了 fused `bias_gelu_bf16` kernel：`add_bias_bf16` + `gelu_inplace` 合一，省一次 DRAM 往返。

- microbench：3.19× 加速
- 落到 captured CUDA Graph：0.7 ms 节省（约 0.5%）

为什么差距这么大？captured graph 早就把 kernel launch overhead 摊干了，剩下的是纯 GPU compute 时间。microbench 的 2-3× 主要来自 Python 启动开销，graph 已经免了。

跨多次类似实验的经验比例：**captured graph 实际节省 ≈ microbench 的 30%**。

下次再看到"我写了个 fusion microbench 快 3×"，先打个 0.3 折扣再看值不值得。

---

## 撞完墙的最终成绩

| 阶段 | p50 | Hz | bit-eq |
|---|---:|---:|:-:|
| 起点 baseline | 128.0 ms | 7.81 | ref |
| + bias_gelu_strict（fusion） | 126.6 ms | 7.90 | ✅ |
| + brln fusion（两对）| 126.4 ms | 7.92 | ✅ |
| + INT8 tile dispatch | **124.4 ms** | **8.04** | ✅ |
| —— 参考：Thor SM110 default frontend 实测 | 46.6 ms | **21.46** | (硬件不同) |

**Orin 累计 +0.23 Hz，全部严格 bit-identical lossless**。Thor 同份 6.8 GB 检查点裸搬过去 default 单 stream 就 21.46 Hz——2.67× Orin，远超 9 Hz 目标。

剩余所有软件杠杆相加 ~10-13 ms 也填不满 9 Hz 的 14 ms 缺口。9 Hz 在 Orin 单卡数学上过不去——要么换硬件（**ThorU SM110 实测 21.46 Hz lossless**，把 6.8 GB 检查点搬过去裸跑 default frontend 没改任何代码就 2.67× Orin），要么接受非严格 lossless（cache_frames=2 已落地，12 Hz at cos=0.991）。

代码全在 [feat/orin-pipelined-streaming](https://github.com/gugudeshubao/FlashRT/tree/feat/orin-pipelined-streaming)，每个 commit 都通过 6 帧固定噪声 maxabs=0 测试。

—— 撞墙日记，写于发现"自己被一次污染的测试结果误导了几个小时"的第二天。
