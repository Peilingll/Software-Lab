import numpy as np
import torch
import torch.nn as nn
import sonata
import open3d as o3d
import plotly.graph_objs as go
import plotly.io as pio

# ===================================================================
# 步驟 1: 加入 SegHead 的類別定義 (從官方 demo 借用)
# ===================================================================
class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)

# ===================================================================
# 主要執行程式碼
# ===================================================================
if __name__ == "__main__":
    # --- 請將這裡的路徑換成您檔案的實際路徑 ---
    npz_path = "tum_main_entrance_processed.npz"
    # -----------------------------------------

    print(f"Loading custom data from: {npz_path}")

    # 載入資料
    point = dict(np.load(npz_path))
    # 預先複製原始座標，用於最後視覺化
    original_coord = point["coord"].copy()

    # 降採樣（避免 GPU 記憶體爆掉）
    num_points_to_keep = 50000
    if len(point["coord"]) > num_points_to_keep:
        print(f"Original number of points: {len(point['coord'])}")
        indices = np.random.choice(len(point["coord"]), num_points_to_keep, replace=False)
        for key in ["coord", "color", "normal"]:
            point[key] = point[key][indices]
        print(f"Down-sampled to {len(point['coord'])} points.")
        # 因為點已經變少，所以用降採樣後的座標做為視覺化的基礎
        original_coord = point["coord"].copy()

    # 載入預設的資料轉換器
    transform = sonata.transform.default()
    point = transform(point)

    # ===================================================================
    # 步驟 2: 載入主模型 (Encoder) 和分類頭模型 (SegHead)
    # ===================================================================
    print("Loading base SONATA model (Encoder)...")
    model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    model.eval() # 設定為評估模式

    print("Loading linear probing SegHead...")
    ckpt = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    seg_head.eval() # 設定為評估模式

    # ===================================================================
    # 步驟 3: 執行推論 (先通過主模型，再通過分類頭)
    # ===================================================================
    print("Running inference...")
    with torch.inference_mode():
        # 將資料放到 GPU
        for k in point:
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda(non_blocking=True)
        
        # 先通過主模型得到特徵 (feat)
        point = model(point)
        feat = point.feat

        # 再將特徵傳給分類頭，得到分類分數 (logits)
        logits = seg_head(point.feat)
        
        # 計算最終預測的類別
        prediction = logits.argmax(dim=-1).cpu().numpy()

    # 若模型在轉換過程中經過了 voxel grid downsample，需要將預測還原回原始解析度
    if hasattr(point, "inverse"):
        print("Mapping predictions back to original resolution...")
        prediction = prediction[point.inverse]

    # ===================================================================
    # 步驟 4: 視覺化結果
    # ===================================================================
    def label_to_color(labels):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        return cmap(labels % 20)[:, :3]

    print("Generating colors for visualization...")
    colors = label_to_color(prediction)

    

    print("Creating Plotly figure...")
    fig = go.Figure(data=[go.Scatter3d(
        x=original_coord[:, 0],
        y=original_coord[:, 1],
        z=original_coord[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=colors,
            opacity=0.9
        )
    )])
    fig.update_layout(
        title="Semantic Segmentation Result (SONATA)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=1))
    )
    pio.renderers.default = 'browser'
    print("Showing figure in browser...")
    fig.show()

    print("Done!")