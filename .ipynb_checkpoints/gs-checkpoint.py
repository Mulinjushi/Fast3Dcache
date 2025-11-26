# import os
# import torch
# import imageio
# from PIL import Image
# from tqdm import tqdm
# import numpy as np

# # --- 环境设置 ---
# os.environ['ATTN_BACKEND'] = 'xformers'
# os.environ['SPCONV_ALGO'] = 'native'

# from trellis.pipelines import TrellisImageTo3DPipeline
# from trellis.utils import render_utils
# from trellis.modules import sparse as sp

# def visualize_gs_stage_evolution(
#     pipeline: TrellisImageTo3DPipeline,
#     image: Image.Image,
#     seed: int = 42,
#     steps: int = 50,
#     cfg_strength: float = 7.5
# ):
#     """
#     从一个固定的相机机位，渲染并保存每个采样步骤的静态法线贴图。
#     """
#     print("开始可视化稀疏结构生成过程 (固定机位，逐帧保存)...")
#     torch.manual_seed(seed)

#     # --- 新增：为保存图片创建输出文件夹 ---
#     output_dir = "render_steps"
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"渲染的图片将保存在 '{output_dir}/' 文件夹中。")

#     print("1. 预处理图像并获取条件...")
#     processed_image = pipeline.preprocess_image(image)
#     cond = pipeline.get_cond([processed_image])

#     print("2. 准备手动循环和固定相机...")
#     flow_model = pipeline.models['sparse_structure_flow_model']
#     sampler = pipeline.sparse_structure_sampler
#     decoder = pipeline.models['sparse_structure_decoder']
    
#     # --- 新增：定义固定的相机参数 ---
#     fixed_yaw = 180.0
#     fixed_pitch = 0.0
#     fixed_r = 2.5
#     fixed_fov = 40.0

#     # --- 新增：在循环外预先计算一次相机矩阵 ---
#     # 注意：render_frames 期望的是一个列表，所以我们传入列表
#     fixed_extrinsics, fixed_intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
#         [fixed_yaw], [fixed_pitch], [fixed_r], [fixed_fov]
#     )
    
#     reso = flow_model.resolution
#     noise = torch.randn(1, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
    
#     t_seq = np.linspace(1, 0, steps + 1)
#     t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
#     x_t = noise

#     print("3. 开始手动执行采样循环并逐帧渲染/保存...")
#     for i, (t_cur, t_prev) in enumerate(tqdm(t_pairs, desc="手动采样中")):

#         with torch.no_grad():
#             out = sampler.sample_once(
#                 model=flow_model, x_t=x_t, t=t_cur, t_prev=t_prev,
#                 cond=cond['cond'], neg_cond=cond['neg_cond'],
#                 cfg_strength=cfg_strength, cfg_interval=[0, 1]
#             )
#         x_t = out.pred_x_prev
        
#         with torch.no_grad():
#             coords = torch.argwhere(decoder(x_t) > 0)
#             if coords.shape[0] == 0:
#                 continue
#             coords = coords[:, [0, 2, 3, 4]].int()

#             slat_feature_dim = pipeline.models['slat_flow_model'].in_channels
#             zero_feats = torch.zeros(coords.shape[0], slat_feature_dim, device=pipeline.device)
#             dummy_slat = sp.SparseTensor(feats=zero_feats, coords=coords)

#             decoded_output = pipeline.decode_slat(dummy_slat, formats=['mesh'])
#             mesh_result = decoded_output['mesh'][0]

#             # --- 核心修改：替换 render_video 为 render_frames ---
#             render_output = render_utils.render_frames(
#                 sample=mesh_result, 
#                 extrinsics=fixed_extrinsics, 
#                 intrinsics=fixed_intrinsics,
#                 options={'resolution': 512, 'bg_color': (0,0,0)},
#                 verbose=False # 在循环中关闭渲染进度条
#             )
            
#             normal_frames = render_output.get('normal', [])
            
#             if normal_frames:
#                 # --- 核心修改：保存单张图片而不是添加到视频列表 ---
#                 frame_array = normal_frames[0] # 获取列表中的第一张（也是唯一一张）图片

#                 # 为保险起见，保留通道顺序修正逻辑
#                 if frame_array.ndim == 3 and frame_array.shape[0] == 3:
#                     frame_array = frame_array.transpose(1, 2, 0)

#                 # 确保形状是 (H, W, C)
#                 if frame_array.ndim == 3 and frame_array.shape[2] in [1, 3, 4]:
#                     # 使用 :03d 格式化文件名，方便排序 (e.g., step_001.png, step_010.png)
#                     output_path = os.path.join(output_dir, f"step_{i+1:03d}.png")
#                     imageio.imwrite(output_path, frame_array)
#                     print(f"\n步骤 {i + 1}/{steps}: 已保存渲染图片到 {output_path}")

#     print("\n🎉 全部步骤渲染完成！")

# if __name__ == '__main__':
#     print("正在加载 Pipeline 模型...")
#     pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
#     pipeline.cuda()

#     print("正在加载图像...")
#     try:
#         image = Image.open("./assets/example_image/typical_creature_rock_monster.png")
#     except FileNotFoundError:
#         print("\n错误：找不到示例图片，请确保 './assets/example_image/typical_creature_rock_monster.png' 路径正确。")
#         exit()

#     visualize_gs_stage_evolution(
#         pipeline,
#         image,
#         seed=42,
#         steps=50,
#         cfg_strength=7.5
#     )

import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

# --- 环境设置 ---
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisImageTo3DPipeline

def save_intermediate_tokens_for_analysis(
    pipeline: TrellisImageTo3DPipeline,
    image: Image.Image,
    seed: int = 42,
    steps: int = 50,
    cfg_strength: float = 7.5
):
    """
    遍历所有去噪步骤，在每一步中捕获并存储所有中间block的特征(tokens)。
    """
    print("开始采集所有步骤和所有中间层的Tokens...")
    torch.manual_seed(seed)

    # 1. 预处理图像并获取条件
    print("1. 预处理图像并获取条件...")
    processed_image = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([processed_image])

    # 2. 准备手动循环
    print("2. 准备手动循环...")
    flow_model = pipeline.models['sparse_structure_flow_model']
    sampler = pipeline.sparse_structure_sampler
    
    reso = flow_model.resolution
    noise = torch.randn(1, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
    
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
    x_t = noise
    
    # 核心数据结构：用一个字典来存储所有数据
    # 结构: {step_index: [block_2_features, block_4_features, ...]}
    all_steps_features = {}

    # 3. 开始手动执行采样循环并保存中间Tokens
    print("3. 开始手动执行采样循环...")
    for i, (t_cur, t_prev) in enumerate(tqdm(t_pairs, desc="正在采集Tokens")):
        
        with torch.no_grad():
            # 获取当前时间步的张量表示
            t_cur_tensor = torch.tensor([1000 * t_cur] * x_t.shape[0], device=x_t.device)
            
            # (A) 调用我们修改后的 forward 函数，直接从当前 x_t 获取中间特征
            _, intermediate_features = flow_model(
                x_t, 
                t_cur_tensor, 
                cond['cond'], # 这里只用正向条件来获取特征
                output_intermediate_features=True
            )
            
            # 将获取到的特征列表存入我们的主字典中
            # .cpu() 是为了将数据从显存转移到内存，防止显存累积
            all_steps_features[i] = [feat.cpu() for feat in intermediate_features]
            
            # (B) 调用采样器，计算下一个时间步的 x_t
            out = sampler.sample_once(
                model=flow_model, x_t=x_t, t=t_cur, t_prev=t_prev,
                cond=cond['cond'], neg_cond=cond['neg_cond'],
                cfg_strength=cfg_strength, cfg_interval=[0, 1]
            )
            x_t = out.pred_x_prev # 更新x_t以进行下一次迭代

    # 4. 将采集到的所有数据保存到文件
    output_path = "intermediate_tokens2.pt"
    print(f"\n4. 所有Tokens采集完成，正在将数据保存到 {output_path}...")
    torch.save(all_steps_features, output_path)
    print("🎉 数据已成功保存！")
    
    # 打印一些信息以供验证
    print("\n--- 数据结构预览 ---")
    print(f"总共保存了 {len(all_steps_features)} 个时间步的数据。")
    first_step_data = all_steps_features[0]
    print(f"第一个时间步包含 {len(first_step_data)} 个中间层的特征。")
    first_step_first_layer_shape = first_step_data[0].shape
    print(f"其中第一个特征张量的形状为: {first_step_first_layer_shape}")


if __name__ == '__main__':
    print("正在加载 Pipeline 模型...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    print("正在加载图像...")
    try:
        image = Image.open("/root/autodl-tmp/TRELLIS_8_5_copy/assets/example_image/plane.png")
    except FileNotFoundError:
        print("\n错误：找不到示例图片，请确保 './assets/example_image/typical_creature_rock_monster.png' 路径正确。")
        exit()

    save_intermediate_tokens_for_analysis(
        pipeline,
        image,
        seed=42,
        steps=50,
        cfg_strength=7.5
    )