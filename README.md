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
