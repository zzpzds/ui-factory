# Reference Test 访问审计

## 结论

Reference v1 的 test split 是模型训练与评测流程的冻结留出集，不是对研究者或标注生成
流程完全不可见的盲测集。Human Gold 与 AI Silver 的构建、标注工作台和包完整性测试均
先于模型正式评测存在，其中 AI Silver 由同一研究工作流生成。因此论文不得声称该 test
对开发者、标注者或 AI 生成流程保持盲法。

固定 `test_unseal.json` 的含义限定为：test IR 首次进入模型训练/评测消费入口的审计
时间。包构建、来源校验和静态完整性审计不产生模型结果，也不得用于模型、阈值、排除
规则或 checkpoint 选择。

## 已知事件

2026-09-01，在实现 Reference Dataset 回归测试时，一个未提交的参数化测试直接构造了
正式 `split="test"` 的 `ReferenceIntentDataset`。根会话执行全量 `pytest` 时，该测试
打开并校验了 20 页 model-ready test IR 及资源哈希，但没有执行模型推理、计算评测指标、
训练参数更新、阈值选择或排除样本，也没有生成正式结果文件。固定
`outputs/intent-reference-v1/test_unseal.json` 未被创建。

不能通过补写 unseal 伪造首次时间。本事件保留为协议偏差，并在正式结果解释中披露。

## 修复

- 删除真实 test Dataset 的参数化构造，只在 train/validation 上做实际加载回归；
- 合成 fixture 只复用开发 split 样本，不复制正式 test reference；
- 在 `ReferenceIntentDataset` 增加正式包 test 门禁，缺少固定 unseal 时在打开 active
  test 资源和 IR 前失败；
- 评测 CLI 在配置、manifest、checkpoint、模型加载和仅元数据的 manifest 语义预检
  成功后创建 unseal，再构造完整 test Dataset；预检前后 manifest hash 必须一致；
- 测试验证未解封门禁不会打开、解析或哈希 active test 文件。

## 论文边界

当前 Reference test 仍可用于训练隔离后的代理一致性评测，但不能支撑“严格双盲测试”
或“独立人工总体真值”的表述。若论文需要严格的未知测试集，应重新抽取未进入当前
Reference 包的新页面，由独立人员保管标签，并在模型与分析方案冻结后单次释放。
