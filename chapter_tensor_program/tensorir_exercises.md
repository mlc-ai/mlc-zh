## TensorIR 练习

```{.python .input n=0}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
import IPython
```

### 第一节：如何编写 TensorIR

在本节中，让我们尝试根据高级指令（例如 Numpy 或 Torch）手动编写 TensorIR。首先，我们给出一个逐位相加函数的例子，来展示我们应该如何编写一个 TensorIR 函数。

#### 示例：逐位相加

首先，让我们尝试使用 Numpy 编写一个逐位相加函数。

```{.python .input n=1}
# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
```

```{.python .input n=2}
# numpy version
c_np = a + b
c_np
```

在我们直接编写 TensorIR 之前，我们应该首先将高级计算抽象（例如，`ndarray + ndarray`）转换为低级 Python 实现（具有元素访问和操作的循环的标准）。

值得注意的是，输出数组（或缓冲区）的初始值并不总是 0。我们需要在我们的实现中编写或初始化它，这对于归约运算符（例如 `matmul` 和 `conv`）很重要。

```{.python .input n=3}
# low-level numpy version
def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
  for i in range(4):
    for j in range(4):
      c[i, j] = a[i, j] + b[i, j]
c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
c_lnumpy
```

现在，让我们更进一步：将低级 NumPy 实现转换为 TensorIR，并将结果与来自 NumPy 的结果进行比较。

```{.python .input n=4}
# TensorIR version
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer[(4, 4), "int64"],
          B: T.Buffer[(4, 4), "int64"],
          C: T.Buffer[(4, 4), "int64"]):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

到这里，我们就完成了 TensorIR 函数。请花点时间完成以下练习。

#### 练习 1：广播加法

请编写一个 TensorIR 函数，将两个数组以广播的方式相加。

```{.python .input n=5}
# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
```

```{.python .input n=6}
# numpy version
c_np = a + b
c_np
```

请完成以下 IRModule `MyAdd` 并运行代码以检查你的实现。

```python
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add():
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

#### 练习 2：二维卷积

然后，让我们尝试做一些具有挑战性的事情：二维卷积。这是图像处理中的常见操作。

这是使用 NCHW 布局的卷积的数学定义：
$$Conv[b, k, i, j] =
    \sum_{di, dj, q} A[b, q, strides * i + di, strides * j + dj] * W[k, q, di, dj],$$
其中，`A` 是输入张量，`W` 是权重张量，`b` 是批次索引，`k` 是输出通道，`i` 和 `j` 是图像高度和宽度的索引，`di` 和 `dj` 是权重的索引，`q` 是输入通道，`strides` 是过滤器窗口的步幅。

在练习中，我们选择了一个小而简单的情况，即 `stride=1, padding=0`。

```{.python .input n=7}
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)
```

```{.python .input n=8}
# torch version
import torch
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch
```

请完成以下 IRModule `MyConv` 并运行代码以检查您的实现。

```python
@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv():
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
```

### 第二节：如何变换 TensorIR

在讲座中，我们了解到 TensorIR 不仅是一种编程语言，而且还是一种程序变换的抽象。在本节中，让我们尝试变换程序。我们在采用了 `bmm_relu` (`batched_matmul_relu`)，这是一种常见于 Transformer 等模型中的操作变体。

#### 并行化、向量化与循环展开

首先，我们介绍一些新的原语：`parallel`、`vectorize` 和 `unroll`。这三个原语被应用于循环上，指示循环应当如何执行。这是示例：

```{.python .input n=9}
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer[(4, 4), "int64"],
          B: T.Buffer[(4, 4), "int64"],
          C: T.Buffer[(4, 4), "int64"]):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
IPython.display.Code(sch.mod.script(), language="python")
```

#### 练习 3：变换批量矩阵乘法程序

现在，让我们回到 `bmm_relu` 练习。首先，让我们看看 `bmm` 的定义：
- $Y_{n, i, j} = \sum_k A_{n, i, k} \times B_{n, k, j}$
- $C_{n, i, j} = \mathbb{relu}(Y_{n,i,j}) = \mathbb{max}(Y_{n, i, j}, 0)$

现在是你为 `bmm_relu` 编写 TensorIR 的时候了。我们提供 lnumpy 函数作为提示：

```{.python .input n=10}
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
```

```python
@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu():
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    # TODO
    ...

sch = tvm.tir.Schedule(MyBmmRelu)
IPython.display.Code(sch.mod.script(), language="python")
# Also please validate your result
```

在本练习中，让我们专注于将原始程序变换为特定目标。请注意，由于硬件不同，目标程序可能不是最好的程序。但是这个练习旨在让你了解如何将程序变换为想要的程序。 这是目标程序：

```{.python .input n=11}
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"], B: T.Buffer[(16, 128, 128), "float32"], C: T.Buffer[(16, 128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))
```

你的任务是将原始程序转换为目标程序。

```python
sch = tvm.tir.Schedule(MyBmmRelu)
# TODO: transformations
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
...

# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)
...

# Step 3. Organize the loops
k0, k1 = sch.split(k, ...)
sch.reorder(...)
sch.compute_at/reverse_compute_at(...)
...

# Step 4. decompose reduction
Y_init = sch.decompose_reduction(Y, ...)
...

# Step 5. vectorize / parallel / unroll
sch.vectorize(...)
sch.parallel(...)
sch.unroll(...)
...

IPython.display.Code(sch.mod.script(), language="python")
```

**（可选）** 如果我们想确保变换后的程序与给定的目标完全相同，我们可以使用 `assert_structural_equal`。请注意，此步骤是本练习中的可选步骤。 如果您将程序**朝着**目标转变并获得性能提升，这就足够了。

```python
tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")
```

#### 构建和评估

最后，我们可以评估变换后的程序的性能。

```python
before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
```
