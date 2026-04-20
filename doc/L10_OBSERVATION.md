# L10 灵心巧手 Observation

根据灵心巧手 Python SDK，`observation` 这里可以只保留 `getStateArc()` 返回的 10 维关节角。

## 推荐定义

- 接口：`getStateArc()`
- 返回类型：长度为 `10` 的浮点数组
- 单位：`rad`
- 含义：当前 L10 灵巧手 10 个可读关节的实时角度

最简写法：

```python
observation = {
    "joint_angles_rad": hand.getStateArc(),  # len = 10
}
```

如果只做策略输入，也可以直接把它当成一个 10 维向量：

```python
joint_angles_rad = hand.getStateArc()  # shape: (10,)
```

## 10 个维度的含义

建议按下面的顺序理解这 10 个维度：

| 维度索引 | 名称 | 含义 |
| --- | --- | --- |
| `0` | `thumb_proximal_bend` | 拇指近端弯曲 |
| `1` | `thumb_distal_coupled_bend` | 拇指远端/联动弯曲 |
| `2` | `index_proximal_bend` | 食指近端弯曲 |
| `3` | `index_distal_coupled_bend` | 食指远端/联动弯曲 |
| `4` | `middle_proximal_bend` | 中指近端弯曲 |
| `5` | `middle_distal_coupled_bend` | 中指远端/联动弯曲 |
| `6` | `ring_proximal_bend` | 无名指近端弯曲 |
| `7` | `ring_distal_coupled_bend` | 无名指远端/联动弯曲 |
| `8` | `pinky_proximal_bend` | 小指近端弯曲 |
| `9` | `pinky_distal_coupled_bend` | 小指远端/联动弯曲 |

## 使用建议

- 训练或推理时，优先使用 `getStateArc()`，因为它直接给出弧度值，最适合作为模型输入。
- 这个 `observation` 就按 `10` 维处理即可，不需要再把温度、电流、触觉等内容混进主 observation。
- 角度范围以设备实际限位和官方文档为准；进入模型前可按需要做归一化。

## 示例

```python
joint_angles_rad = hand.getStateArc()

assert len(joint_angles_rad) == 10

thumb = joint_angles_rad[0:2]
index = joint_angles_rad[2:4]
middle = joint_angles_rad[4:6]
ring = joint_angles_rad[6:8]
pinky = joint_angles_rad[8:10]
```

参考文档：[Linkerbot Python SDK 安装页](https://docs.linkerhub.work/sdk/zh-cn/)
