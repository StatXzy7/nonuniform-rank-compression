# 非均匀低秩矩阵压缩引擎

这是一个只依赖 JDK 标准库的 Linux CPU Java 实现。输入矩阵每维不超过 12，批量中的每个矩阵都按自身最大奇异值归一化，再用同一个阈值决定保留方向。

## 构建与运行

要求 OpenJDK 17 或更高版本，不需要网络、Maven、Gradle 或第三方矩阵库：

```bash
bash ./verify.sh
```

脚本会编译主程序和测试、执行非零数量的自动化检查，并运行一个有限小型示例。离线环境只需预装 JDK；没有需要额外下载或缓存的依赖。

## 压缩规则

- 每个矩阵使用薄 SVD，薄谱方向总数为 `min(rows, columns)`，零奇异方向始终计入删除分母并删除。
- 非零奇异值除以本矩阵最大奇异值；批内共用一个阈值。
- 直接阈值模式：归一化值严格小于阈值才删除；与阈值相等的非零方向保留。
- 目标删除比例模式：分母是整批薄谱方向总数，包含零奇异方向；枚举由相同谱值形成的离散可达结果，选择偏差最小的预算。等距时优先少删，不会拆开相同谱值。
- SVD 的任意向量符号或重根基底不影响保留秩、存储决策、重建和乘法。
- 因子形式为 `UΣ` 与 `Vᵀ`，秩为 `r` 时因子元素数是 `r*(m+n)`。
- 只有 `r*(m+n) < m*n` 时才实际保存因子；否则保存截断后的稠密矩阵。报告同时给出谱截断秩和真实存储节省，二者不混同。
- 全零矩阵恒为零秩，所有薄方向删除；重建误差为零，不进行除零。
- 空矩阵、重复名称、不在 `[0,1]` 或非有限的阈值/目标比例、非有限矩阵元素都会明确抛出 `IllegalArgumentException`。

## 持久化与乘法

`CompressionResult.save(path, result)` 先写同目录临时文件，再写入 SHA-256 载荷摘要和长度前缀数据，完整后原子移动到目标。`load` 会验证签名、版本、长度、摘要、维度、秩和有限值；损坏文件会被拒绝，保存失败不会替换已有的完整结果。

矩阵乘法在重建后的稠密矩阵上执行，因此保存前后的重建和乘法结果逐 bit 一致。

## 核心 API

```java
NamedMatrix a = new NamedMatrix("a", new Matrix(2, 2, 1, 0, 0, 0.5));

CompressionResult byThreshold = CompressionResult.compressByThreshold(0.1, a);
CompressionResult byBudget = CompressionResult.compressByTargetFraction(0.5, a);

CompressedMatrix compressed = byThreshold.matrix("a");
int rank = compressed.retainedRank();
double relativeError = compressed.relativeReconstructionError();
int factorElements = compressed.factorElementCount();
int storedElements = compressed.storedElementCount();
Matrix reconstructed = compressed.reconstruct();

CompressionResult.save(Path.of("batch.nrcb"), byBudget);
CompressionResult loaded = CompressionResult.load(Path.of("batch.nrcb"));
Matrix product = loaded.multiply("left", "right");
```

## 数值说明

实现使用单侧循环 Jacobi SVD。对输入 Frobenius 范数量级为零附近的数值奇异值，采用 `8*ulp(scale)*min(m,n)` 识别为数值零；归一化谱值比较使用 `1e-12` 作为“相同/等阈值”容差，避免 SVD 舍入把数学上相同的谱值拆开。
