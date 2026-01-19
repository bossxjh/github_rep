#输入模型名称，标准格式的数据集，一个视频提取多少帧？是需要多样性还是可学性？（如果不指定模型和帧，就默认是用clip模型来提取3帧，默认输出多样性）
#输入超参数。【默认超参数由超参数敏感性分析实验得到】
#是否要进行数据的预处理？【质量分析，是否都是成功案例，筛选一遍等。默认不处理】
#加载模型对应的视觉头，并且提取数据集特征【默认只包含图像输入，还可以选择加入state和task特征进行融合】
#用数据集特征得到数据集多样性和可学性

#暂定的模型：clip，openvla，pi0（pi0.5）,diffusion policy
#支持的数据集有：
#标准格式：如果要增加数据：libero/robotwin的格式
#数据预处理？（其实没必要，待定吧）
#融合模块？

#指标计算的代码实现
#主要是SRCC,PLCC,KRCC三者和真值之间的相关性

#算法的对比：就是用一些之前的简单的传统方法，来计算指标，然后和我们的方法进行对比.【有10个，或者5个就好】


# scripts/main.py
# python -m scripts.main
# from dataeval.api import extract_features

# if __name__ == "__main__":
#     dataset_path = "/Volumes/T7/数据集/具身/libero_100/libero_10"
#     for feat in extract_features("diffusion policy", "libero", dataset_path, num_frames=1,batch_size=16):
#         print(feat.shape)


#下一步为了可以得到科学性指标，我的特征里除了：1，样本特征矩阵，需要（N，D）维度的  2，还需要任务长度task_lengths字典 3，还需要任务描述文本task_descriptions字典 4，还需要任务类别task_categories字典 


# from dataeval.api import extract_features_with_metadata

# if __name__ == "__main__":
#     dataset_path = "/Volumes/T7/数据集/具身/libero_100/libero_10"
#     demo_features = extract_features_with_metadata("diffusion policy", "libero", dataset_path, num_frames=3, batch_size=16)
#     print(len(demo_features))         # demo 数量
#     print(demo_features[0].keys())    # ['task_id', 'task_length', 'demo_length', 'task_description', 'features']
#     print(demo_features[0]['features'].shape)  # 模型特征形状

# from dataeval.api import extract_features_with_metadata
# import os
# import numpy as np
# from tqdm import tqdm
# import time

# if __name__ == "__main__":
#     dataset_path = "/Volumes/T7/数据集/具身/libero_spatial"
#     save_dir = "/Volumes/T7/项目/具身数据评测/github_rep/feature"
#     os.makedirs(save_dir, exist_ok=True)

#     # 配置
#     model_name = "diffusion policy"
#     dataset_name = "libero"
#     num_frames = 3
#     batch_size = 16

#     start_time = time.time()

#     # 提取特征
#     demo_features = []
#     for feat_dict in tqdm(
#         extract_features_with_metadata(model_name, dataset_name, dataset_path,
#                                        num_frames=num_frames, batch_size=batch_size),
#         desc="Extracting features"
#     ):
#         demo_features.append(feat_dict)

#     total_time = time.time() - start_time

#     # 打印信息
#     print(f"demo 数量: {len(demo_features)}")
#     print(f"demo 字典 keys: {demo_features[0].keys()}")
#     print(f"第一个 demo 特征 shape: {demo_features[0]['features'].shape}")
#     print(f"总耗时: {total_time:.2f} 秒")

#     # 准备保存
#     features_array = np.stack([d["features"] for d in demo_features], axis=0)
#     task_ids = np.array([d["task_id"] for d in demo_features])
#     task_lengths = np.array([d["task_length"] for d in demo_features])
#     demo_lengths = np.array([d["demo_length"] for d in demo_features])
#     task_descriptions = np.array([d["task_description"] for d in demo_features], dtype=object)

#     # 保存文件名包含配置信息
#     save_name = f"{dataset_name}_{model_name.replace(' ', '_')}_nf{num_frames}_bs{batch_size}.npz"
#     save_path = os.path.join(save_dir, save_name)

#     np.savez_compressed(
#         save_path,
#         features=features_array,
#         task_ids=task_ids,
#         task_lengths=task_lengths,
#         demo_lengths=demo_lengths,
#         task_descriptions=task_descriptions
#     )
#     print(f"保存完毕: {save_path}")




# test_leanability.py
import numpy as np
from dataeval.metric.leanability import compute_leanability_from_npzdata

def main():
    npz_path = "/Volumes/T7/项目/具身数据评测/github_rep/feature/libero_spatial_clip_nf3_bs16.npz"  # 改成你的 npz 路径

    npzdata = np.load(npz_path, allow_pickle=True)

    result = compute_leanability_from_npzdata(
        npzdata=npzdata,
        beta=0.5,
        sigma=0.001,
    )

    print("\n===== Leanability Result =====")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()




#然后就是需要得到评测的代码，记录gt和预测值，然后计算SRCC,PLCC,KRCC等指标

# 还可以加上结果的可视化
# ( 94.00,  0.184274)
# ( 96.00,  0.178297)
# ( 76.00,  0.177805)
# ( 96.00,  0.176177)
# ( 82.00,  0.178632)
# ( 84.00,  0.177643)
# ( 88.00,  0.181925)
# ( 90.00,  0.180233)
# ( 80.00,  0.176968)
# ( 72.00,  0.177289)

# {0: 0.1847720561023648, 1: 0.17871149872377087, 2: 0.17809390392050242, 3: 0.1764156005836858, 4: 0.17873688943489277, 5: 0.17779078324272843, 6: 0.18140676669754277, 7: 0.18069241156417995, 8: 0.17705977762717315, 9: 0.17766571079174934}


