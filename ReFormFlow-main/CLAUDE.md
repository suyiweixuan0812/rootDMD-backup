# ReFormDMD

## 范围

- 源码基线：ReFormFlow-release `f9b9e5da44c20fae347500af83dccefb43f00cba`。
- 保留 ReFormFlow SD3/FLUX RL；新增 SD3/SD3.5 标准 DMD 与 `ac_dmd.enabled` 控制的 AC-DMD。
- DMD 初版仅 PEFT LoRA、DDP；不包括 FSDP、FLUX DMD、量化或 RL/DMD 联合目标。
- `../ReFormFlow-release`、`../FlowAWR`、`../RTDMD` 和旧 `../RDMD` 不作为可修改目标。

## 文件职责

- `scripts/train_dmd_sd3.py`：独立保留 `dmd_loss`、`fake_loss`、`ac_loss` 和 `save_ckpt`；初始化、采样、角色切换、训练、EMA、评估、恢复及保存时机直接写在 `main`，检查点写入由 `save_ckpt` 执行。不要引入 trainer、模型包装类或预测工厂。DDP 直接包装共享冻结 Transformer 及 `generator` / `fake_score` 两套 LoRA，Teacher 通过禁用 adapter 执行。
- `flow_grpo/train_utils.py`：通用分布式初始化/清理、主进程判断、随机种子与 RNG 状态、DDP 解包和广播工具；DMD 脚本直接导入，不在脚本内重复定义。不包含 DMD 损失、adapter 或检查点逻辑。
- `flow_grpo/prompt_data.py`：SD3 ReFormFlow 与 DMD 共用 Dataset 和 collate，RL 的 K-repeat sampler 原规则不变。DMD 在训练脚本中单独使用 PyTorch DistributedSampler，按数据 epoch 洗牌、连续取批；sampler 和 DataLoader 均 drop_last=True，同一数据 epoch 内跨卡、跨批索引不重复，不按文本内容去重。同一外层迭代两个角色复用条件。
- `config/dmd.py`：DMD 参数及 SD3.5-Medium/SD3-Medium 预设；继承 base.py 并复用 sample/train/顶层字段，不修改 base.py 或 reform.py 的 RL 默认行为。sample 中配置 denoising_step_list、num_steps、random_stop_idx、noise_level、test_batch_size；训练和评估共用列表前 num_steps 项生成的时间表，不额外使用 sample.eval_num_steps 或 sample.guidance_scale。train.learning_rate/fake_learning_rate 分别用于两角色，其余 Adam 参数、max_grad_norm、warmup_steps 共用，EMA 开关为 train.ema。num_epochs 是外层训练轮数而非 data_epoch；保存/评估间隔为 save_freq/eval_freq。dmd/ac_dmd 仅保留对应算法参数，不恢复 solver 配置字典或无效的 ac_dmd.start_mode。
- `flow_grpo/diffusers_patch/sd3_pipeline_dmd.py`：只保留直接调用 Transformer 的 `generate_latents`，控制选中末步的梯度并返回 `(x0, x_s)`；每次模型前向独立进入、退出 autocast，不传预测回调，不包含 loss、backward 或 optimizer。
- `flow_grpo/diffusers_patch/solver.py`：CPS 单步数值更新放在 `cps_step`，保持 FP32 中间计算；不得为 CPS 添加全局 no_grad，不改变既有 RL solver 分支。DMD 的 CFG、时间表与 AC 加噪/目标在训练脚本直接展开，文本编码复用原始 `encode_prompt`。
- `flow_grpo/rewards.py` 和 scorer 保留来源接口与目录，不改成 `rewards/` 包。
- `jobs/`：本地环境、资源检查和启动。禁止写入凭据、自动安装或删除用户缓存。
- `scripts/check_runtime.py`：无权重资源检查，不代表真实评分/CUDA 已通过。

## 环境与资源

- 使用 `../miniconda3/envs/FlowAWR`；可以通过 `REFORMDMD_ENV` 覆盖。
- `jobs/local_env.sh` 优先使用目标工程的 Python 路径；检查实际导入来源，不依赖旧 FlowAWR 可编辑安装回退。
- `reward_ckpts` 是共享目录链接；OCR 使用用户的共享缓存链接。已有冲突路径不得覆盖。
- 原 `setup.py` 为来源依赖声明，不据此重装或降级现有环境。

## 验证

- 当前不维护 `tests/` 目录，不随训练脚本修改而重建或同步测试包装。以 `scripts/train_dmd_sd3.py` 的实际训练行为为准，验证尽量集中到阶段验收；历史结果不代表当前版本通过。
- 真实权重加载、GPU 训练、下载、安装、提交和推送须另行获得授权。
- CPU 小模型测试不证明真实多卡显存、CUDA 扩展或全部奖励兼容。
- 保留标准 DMD 和参考 AC-DMD 的数值行为；算法调整需要单独提出。
- DMD 默认关闭 gradient checkpointing，通过 `config/dmd.py` 覆盖，不修改公共 `config/base.py` 中的 RL 默认值。
- 推理 LoRA 与训练恢复状态合并在 `checkpoint-N/` 下：`lora/generator/` 与 `lora/fake_score/` 是两套 adapter，`optimizer.pt`、`scaler.pt` 供恢复，`config.json` 记录配置；步数由目录名解析。保存时以 Generator EMA 权重写盘，随后恢复训练参数。不保存数据迭代器或训练随机数状态；以 `divmod(outer_step, len(train_dataloader))` 推导数据 epoch 和批次偏移，设置 sampler.set_epoch(data_epoch)，重建迭代器并跳过已消费批次。数据 epoch 边界才重新创建迭代器；每轮在迭代器创建及恢复定位之后使用 `seed + outer_step * world_size + rank` 设置训练种子。该策略要求相同数据及顺序、seed、卡数和 batch，且在外层迭代结束处保存。旧 KRepeat 检查点可继续训练，但不重放旧抽样轨迹。Generator 仍从导出的 EMA 权重恢复，scheduler 和 EMA 历史不恢复，不等价于完整连续优化状态。只能加载可信本地训练检查点。
- 两套 adapter 按 AWR 方式分别激活后筛选 `requires_grad` 参数，并在 DDP 构造时共同开启梯度以登记；不能照搬仅优化一套 adapter 的登记方式或解冻整个基座。训练循环显式使用 `set_adapter`、`no_grad` 和 `disable_adapter`，临时目标预测后切回当前训练角色再 backward，不在待重算的计算图中更换权重。采用 cold-start 逐批更新；当前入口不读取或校验 `gradient_accumulation_steps`，不使用跨批 `conditions` 缓存或 `no_sync` 累积；Generator 更新直接复用当前批文本编码并独立采样。
- LoRA 结构在训练脚本直接固定为 r=32、alpha=64、Gaussian 初始化，以及 AWR 的八类 `attn.*` 投影；SD3/SD3.5 共用，不扩展到 SD3.5 的 `attn2`。`config.lora` 仅保留 generator_path/fake_score_path；warm-start 和 resume 使用同基座、符合该固定结构的可信检查点，不另行比对 LoRA 配置字典。
- adapter 切换使用 PEFT 0.10.0 的 `set_adapter` 自带梯度标志处理，不依赖新版 PEFT 的状态 API。warm-start 必须先注册两套 adapter、转 FP32，再加载权重；旧 PEFT 添加 adapter 会按基座 dtype 转换整个 LoRA 层，不能在第二套 adapter 注册前加载第一套权重。

<history_timeline>

@.claude/history/timeline_view.md

</history_timeline>

<project_structure>

@.claude/project_tree.md

</project_structure>

<logic_tree>

@.claude/logic_tree_view.md

</logic_tree>
