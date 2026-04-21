# UniDex L10 特调版

UniDex-VLA 动作生成部分的一份简化说明：模型同时接收点云、语言指令和机器人自身状态，在统一的 FAAS 动作空间中预测未来一段动作。

## 模型输入

模型在时刻 `t` 的输入由三部分组成：

- `Point cloud`：由 `Uni3D` 编码，再通过 projector 对齐到 `PaliGemma/Gemma` 的 hidden size。
- `Language instruction`：直接使用 `PaliGemma/Gemma` 的文本 embedding。
[!NOTE]PaliGemma/Gemma需要微调
- `Proprioception`：机器人当前本体状态，通过 `Linear(proprio_dim=82 -> hidden=1024)` 投影到隐藏空间。

## 动作是怎么建模的

UniDex-VLA 不是直接预测单步动作，而是一次预测未来 `H` 步组成的 `action chunk`：

`A_t = [a_t, ..., a_{t+H-1}]`

训练时，模型不会直接看干净动作 `A_t`，而是先构造一个带噪版本：

`A_t^τ = τ A_t + (1 - τ) ε, 其中 ε ~ N(0, I), τ ∈ [0, 1]`

这里的直觉是：让模型学会如何把一段“被噪声污染过的未来动作”逐步拉回真实动作轨迹，而不是一次性硬回归出最终答案。

具体来说：

- `ActionEncoder(action_dim=82 -> hidden=1024)` 先把 FAAS 空间中的动作表示编码成动作 token。
- `timestep embedding` 用来注入当前时间步/噪声强度 `τ`，告诉模型现在处在去噪过程的哪个阶段。
- 当前观测 `o_t = [P_t, l_t, q_t]` 和带噪动作块 `A_t^τ` 一起送入主干网络，学习条件向量场 `v_θ(A_t^τ, o_t)`。
- 监督目标不是直接回归动作值，而是逼近论文中的去噪方向 `u(A_t^τ | A_t) = A_t - ε`。

## 主干网络与输出

上面的多模态 token 和动作 token 会一起进入 `Joint Transformer / mixture model`，得到动作相关的隐藏状态 `action hidden states`。

最后，模型通过一层线性投影：

`Linear(hidden=1024 -> action_dim=82)`

把隐藏状态映射回 FAAS 动作空间，输出对应的 `action velocity`。

训练过程

```mermaid
flowchart TD
    A[RGB-D / Colored Pointcloud P_t] --> B[Uni3D Pointcloud Encoder]
    B --> B1[Pointcloud Tokens / 3D Visual Embedding]

    C[Language Prompt<br/>例如: grasp the cup] --> D[PaliGemma Tokenizer + Text Embedding]
    D --> D1[Language Tokens]

    E[Current Proprioception s_t<br/>wrist pose + hand joint state<br/>FAAS-like 82D] --> F[Proprio Encoder<br/>Linear / MLP]
    F --> F1[Proprio Tokens]

    G[Ground-truth Future Action Chunk<br/>shape H x 82] --> H[Sample Gaussian Noise<br/>same shape H x 82]
    H --> I[Sample diffusion time]
    G --> I
    I --> J[Construct Noisy Action Chunk<br/>interpolate clean action and noise]

    J --> K[Action Encoder<br/>Noisy action tokens]
    I --> K1[Time Embedding]
    K1 --> K

    B1 --> L[Joint Transformer / Mixture Backbone]
    D1 --> L
    F1 --> L
    K --> L

    L --> M[Action Expert Hidden States]
    M --> N[Action Decoder / Linear Head]
    N --> O[Predicted Vector Field<br/>shape H x 82]

    G --> P[Target Vector Field<br/>denoising direction]
    H --> P

    O --> Q[Flow-Matching Loss<br/>match predicted and target field]
    P --> Q
```



推理过程

```mermaid
flowchart TD
    A[RGB-D / Colored Pointcloud P_t] --> B[Uni3D Pointcloud Encoder]
    B --> B1[Pointcloud Tokens]

    C[Language Prompt] --> D[PaliGemma Tokenizer + Text Embedding]
    D --> D1[Language Tokens]

    E[Current Proprioception s_t<br/>current wrist + hand state] --> F[Proprio Encoder]
    F --> F1[Proprio Tokens]

    R[Sample Initial Gaussian Action Chunk<br/>shape H x 82] --> S[Current Action Chunk]

    S --> T[Action Encoder]
    U[Current Euler Step] --> U1[Time Embedding]
    U1 --> T

    B1 --> V[Joint Transformer / Mixture Backbone]
    D1 --> V
    F1 --> V
    T --> V

    V --> W[Action Expert Hidden States]
    W --> X[Action Decoder]
    X --> Y[Predicted Vector Field]

    Y --> Z[Euler Update<br/>advance action chunk one step]
    Z --> S

    Z --> AA[After N steps<br/>Denoised Action Chunk]
    AA --> AB[FAAS Action Chunk<br/>H x 82]
    AB --> AC[Take first action / first few actions]
    AC --> AD[FAAS-to-Robot Adapter]
    AD --> AE[Arm Command + Dexterous Hand Command]
```



总体过程

```mermaid
flowchart LR
    A[Pointcloud] --> A1[3D Visual Tokens<br/>Uni3D]
    B[Prompt] --> B1[Text Tokens<br/>PaliGemma/Gemma]
    C[Proprioception] --> C1[State Tokens<br/>Linear/MLP]
    D[Noisy Action Chunk + t] --> D1[Action Tokens<br/>Action Encoder + Time Embedding]

    A1 --> E[Joint Transformer / Mixture Model]
    B1 --> E
    C1 --> E
    D1 --> E

    E --> F[Updated Action Tokens]
    F --> G[Action Decoder]
    G --> H[Vector Field / Action Velocity]
```



对于该项目

```mermaid
flowchart TD
    A[Policy Output<br/>H x 82 FAAS Action Chunk] --> B[Select first step / first K steps]
    B --> C[Right Wrist Pose Delta]
    B --> D[Right Hand FAAS Joint Slots]
    C --> E[6-DoF Arm Controller<br/>IK / Cartesian Servo]
    D --> F[L10 FAAS Adapter<br/>82D -> 10 active joints]
    E --> G[Robot Arm Motion]
    F --> H[L10 Finger Motion]
```



