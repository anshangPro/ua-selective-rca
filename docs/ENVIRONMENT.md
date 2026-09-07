# 已验证环境（RE1 第 1 周）

- Python：3.8.10，项目虚拟环境为 `.venv`。
- 数据：官方 Hugging Face 数据集 `phamquiluan/RCAEval` 的 `re1*` 快照，目录为 `data/RCAEval`；RE1-OB、RE1-SS、RE1-TT 各 125 个案例。
- 上游代码：RCAEval `bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90`；BARO `e35f4ec1095e5cac891d52de9ad18a5b32a37ec8`。
- 已验证基线：在 `re1ss_carts_cpu_1`（以 `inject_time.txt` 为切分边界）上，BARO 与 CIRCA 均返回 52 项指标排名；CPU 耗时分别为约 0.03 s 和 4.33 s。

## RCD 说明

RCAEval 的 RCD 锁文件要求其仓库 `lib/causallearn` 中的定制
`SkeletonDiscovery.py`。该补丁与 CIRCA 所需的 `causal-learn` API 版本不兼容：
旧版缺少 CIRCA 的 `node_names` 参数，较新版则改变了 RCD 补丁调用的
`CausalGraph` 构造函数。Windows 下完整 RCD extras 还会在不被 RCD 使用的
`tigramite` OpenMP 扩展编译失败。RCD 结果在此问题消除前不得纳入实验对比；
应使用隔离的、由上游锁文件和链接补丁完全验证的 RCD 环境。
