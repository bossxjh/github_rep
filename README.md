# github_rep

feature_extractor/
│
├── models/
│   ├── clip.py
│   ├── dinov2.py
│   ├── siglip.py
│   ├── xclip.py
│   └── registry.py
│
├── utils/
│   ├── sampling.py
│   └── io.py
│
├── extractor.py   ← 最核心的用户接口文件（别人主要只调用这个）
└── __init__.py


## Developer Notes
dataeval/
├── api.py          ← 对外统一接口（最重要）
├── sampling.py     ← 数据采样策略
├── models.py       ← 模型注册 / 适配
├── models/
│   └── openvla.py  ← OpenVLA 具体实现
├── datasets/       ← 各种机器人数据集适配
│   ├── libero.py
│   ├── bcz.py
│   ├── taco_play.py
│   ├── jaco_play.py
│   └── ...
├── config.py       ← 全局配置


运行测试：python -m scripts.main