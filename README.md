# 多层模型伪代码
分为
- 模型基类（负责初始化过程和训练监控）
- 残差层（实现接口）
- 单层残差模型（验证模型效果）
- 多层残差模型（最终实现）

## 〇、模型基类 Layer
所有层和模型的基类，负责模型(层)固有的属性，并负责训练监控
不负责 train 策略和 solve 求解参数过程
规定子类模型必须实现的函数，规定部分函数必须执行的操作

### 0.1 固有参数 `set_layer`
初始化后固定不变的参数，表征模型的固有特征，决定了前向过程和训练过程
- idx (layer_idx)
- act (torch.relu)
- device ("cuda:0")

### 0.2 数据形状参数 `set_shape`
- 方法一：传入 m, d, k
- 方法二：使用 **kwargs 传递字典

决定了 0.3 模型参数的维度, 并与 0.6 训练数据的形状一致

### 0.3 模型参数 `set_self`
**子类必须实现 `set_self()`**
- `self.A`
- `self.W`
- `self.A_l`
- `self.W_l`
- `self.head` 应在初始化过程中绑定，因为 solve 要用到

**以上 0.1 - 0.3 由 init 调用，init 也必须实现这些操作**

### 0.4 规定前向过程
- **子类必须实现 `forward`**
- 基类隐式实现 `__call__`

### 0.5 训练监控参数 `set_check`
只在训练监控中起作用的参数，初始化后只能由`_check_conv`更改
- loss
- acc
- conv

### 0.6 训练数据参数 `set_data`
模型暂存它们用于监控训练过程，由于 Python 的赋值操作是浅拷贝，因此不会产生额外内存开销
- feature_map，可以是以下值
    - X
    - Z_prev
    - Z
- label，只能是以下值
    - Y_true
- n
    - label.shape[0]

### 0.7 训练监控方法
计算 loss 值 `test`
- Input：X，Y
- Inter：调用 `forward` 方法
- Output：loss(l2_loss), acc

检查收敛性 `_check_conv`
- 只能在类内部 train 过程中使用
- Input：通常由 `set_data` 指定 X，Y
- Inter：
    - 调用 `test` 方法
    - 更新 loss，acc，conv
- Output：conv

**以上 0.5 - 0.7 由 train 调用**


### 0.8 函数传入的参数
不保存在模型中，作用域只在调用函数内部的被传参数
- max_iter
- Y_res

### 0.9 规定子类必须实现的函数
- `set_self`
- `forward`

### 0.10 规定部分函数必须执行的操作
- `__init__` 必须调用
    - set_layer
    - set_shape
    - set_self
- `train` 必须调用
    - set_check
    - set_data
    - _check_conv
- `train` 中还可以调用 test


## 一、残差层 ResidualLayer
### 1. 数学公式
**优化目标：**
$$
\begin{aligned}
L &= \min_{A^l, W^l} \left\| A(Z^{l-1} + A^l |Z^{l-1} W^l|_+) W - Y \right\|_F^2 \\
&= \min_{A^l, W^l} \left\| A A^l |Z^{l-1} W^l|_+ W - Y_{res} \right\|_F^2 \\
\end{aligned}
$$
**变量定义：**
- $Z^{l-1}$：输入特征图  
- $Y$：真实标签（已知）
- $Y_{res} = Y - Y^{l-1} = Y - A Z^{l-1} W$: 拟合目标
- $A$：分类头内池化层，$W$：线性层（已知）
- **待优化参数**：$A^l$, $W^l$（未知）

**变量名映射：**
inputs
- $A$ → `head.A`
- $W$ → `head.W`
- $Z^{l-1}$ → `Z_prev`
- $Y$ → `Y_true`
- $Y^{l-1}$ → `Y_prev`
- $Y_{res}$ → `Y_res`

paras
- $A^l$ → `A_l`
- $W^l$ → `W_l`

output
- $Z^l$ → `Z`


### 2. 模型结构
初始化 `init`
- 按基类 `Layer` 规定，应调用`set_layer`,`set_shape`, `set_self`
- 并且，由于中间层绑定分类头，在初始化时要传入分类头引用 `self.head`


### 3. 前向过程
按基类 `Layer` 规定，必须实现 `forward`
- 提取特征图 `forward_feature()`
    - input: `Z_prev`
    - output: `Z = Z_prev + A_l | Z_prev W_l|_+`
- 计算预测值 `forward_with_head()`
    - input: `Z_prev`
    - inter: `Z = forward_feature()`
    - output: `Y_pred = A Z W`
- 令 `forward` 为 `forward_feature()`

### 4. 训练过程 *(Optional)*
- 使用真实标签 Y_true 进行模型训练 `train_label`
    - input: `Z_prev`, `Y_true`
    - inter: 
    `self.Y_res = Y_true - head.forward(Z_prev)`
    `A_l = solve_A_l`
    `W_l = solve_W_l`
    - check: `_check_conv`
    - No output
- 使用残差值 Y_res 进行模型训练 `train_res`
    - input: `Z_prev`, `Y_res`
    - inter: 
    `A_l = solve_A_l`
    `W_l = solve_W_l`
    - cann't check
    - No output

4. 参数更新方法
- `solve_A_l`
    - input: `head.A`, `head.W`, `Z_prev`, `Y_res`
- `solve_W_l`
    - input: `head.A`, `head.W`, `Z_prev`, `Y_res`

## 二、单层残差模型 ResidualLayerModel
### 1. 数学公式
**优化目标：**
$$
\begin{aligned}
L &= \min_{A^1, W^1} \left\| A(X + A^1 |X W^1|_+) W - Y \right\|_F^2 \\
&= \min_{A^1, W^1} \left\| A A^1 |X W^1|_+ W - Y_{res} \right\|_F^2 \\
\end{aligned}
$$
**变量定义：**
- $X$：输入特征图（已知）
- $Y$：真实标签（已知）
- $Y^0 = A X W$: 中间变量，暂未使用
- $Y_{res} = Y - Y^0$: `ResidualLayer`拟合目标
- $Z = X + A^1 |X W^1|_+$: 残差连接后的特征图

**待优化参数**
- 分类头 `head`
    - $A$：分类头池化层
    - $W$：分类头线性层
- 残差层 `block`
    - $A^1$：残差层Patch信息
    - $W^1$：残差层Channel信息

**变量名映射：**
inputs
- $X$ → `X`
- $Y$ → `Y`

paras
- $A$ → `head.A`
- $W$ → `head.W`
- $A^1$ → `block.A_l`
- $W^1$ → `block.W_l`

inter
- $Y^0$ → `Y_prev`
- $Y_{res}$ → `Y_res`
- $Z$ → `Z`


### 2. 模型结构
- 初始化 `init`
    - `self.head`、`self.block`

### 3. 前向过程
- 完整前向过程 `forward`
    - input: `X（data）`
    - inter: `Z = block.forward_feature(X)`
    - output: `Y_pred = head.forward(Z)`
- 计算 loss 和 acc 值 `test`
    - input *(Optional)*: `X`, `Y`
    - inter: `Y_pred = forward()`
    - output: `loss, acc`
- 对比 loss 和 acc 差异 `_check_conv`
    - input *(Optional)*: `X`, `Y_true`
    - inter: `loss, acc = test()`
    - output: `conv`

3. 训练过程
- 外层训练 `train_outer`
    - input：`X`, `Y_true`
    - inter：`head.train()`, `block.train()`
    - No output
- 调用内层训练 `train_lin`
    - input：`X`, `Y_true`
    - inter: 
        ```python
        head.W = head.solve_W()
        head.A = head.solve_A()
        block.A_l = block.solve_A_l()
        block.W_l = block.solve_W_l()
        ```
    - No output

## 三、多层残差模型 MultiResidualLayerModel
### 3.1 定义接口
1) 模型结构
- 初始化 `init`
    - head: `ClassifierHead`
    - blocks: `list[ResidualLayer]`

2) 前向过程
- 获取特征图 `forward_features`
- 完整前向过程 `forward`
- 计算 loss 和 acc 值 `test`
- 对比 loss 和 acc 差异 `_check_conv`

3. 训练策略
- 逐层收敛训练 `train_list`
    - 调用 head.train
    - 调用 layer.train
- 调用内层训练 `train_lin`

#
#
#

## 五、进阶模型，不使用 $Z^0 = X$
