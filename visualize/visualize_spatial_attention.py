import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ----------------------
# 读取两个文件中的 bbox
# ----------------------
file1 = "/home/hang18/links/projects/rrg-vislearn/hang18/bboxes0.txt"
file2 = "/home/hang18/links/projects/rrg-vislearn/hang18/bboxes1.txt"

# np.loadtxt 读取文件，确保 shape 为 (N,4)
bboxes1 = np.loadtxt(file1)  # [N1,4]
bboxes2 = np.loadtxt(file2)  # [N2,4]

print('len: ', len(bboxes1))

assert bboxes1.shape[0] == bboxes2.shape[0], "两个文件的bbox数量不一致"

N = bboxes1.shape[0]
# 随机生成 0/1，决定是否交换
swap_flags = np.random.rand(N) < 0.5  # 50% 概率交换

for i in range(N):
    if swap_flags[i]:
        bboxes1[i], bboxes2[i] = bboxes2[i].copy(), bboxes1[i].copy()  # 交换

bboxes1_resized = []
bboxes2_resized = []

for b1, b2 in zip(bboxes1, bboxes2):
    # bbox1 resize
    x1, y1, x2, y2 = b1
    cx1, cy1 = (x1 + x2)/2, (y1 + y2)/2
    w1, h1 = x2 - x1, y2 - y1
    scale_x, scale_y = 1.0 / w1, 1.0 / h1
    # print(scale_x, scale_y)

    new_b1 = [0, 0, 1, 1]
    bboxes1_resized.append(new_b1)

    bx1, by1, bx2, by2 = b2
    cx2, cy2 = (bx1 + bx2)/2, (by1 + by2)/2
    delta_cx, delta_cy = cx2 - cx1, cy2 - cy1
    new_cx2, new_cy2 = 0.5 + delta_cx * scale_x, 0.5 + delta_cy * scale_y
    w2, h2 = bx2 - bx1, by2 - by1
    new_w2, new_h2 = w2 * scale_x, h2 * scale_y
    new_b2 = [new_cx2 - new_w2/2, new_cy2 - new_h2/2, new_cx2 + new_w2/2, new_cy2 + new_h2/2]
    bboxes2_resized.append(new_b2)

bboxes1_resized = np.array(bboxes1_resized)
bboxes2_resized = np.array(bboxes2_resized)

# all_bboxes = np.vstack([bboxes1_resized, bboxes2_resized])
bins = 100
heatmap = np.zeros((bins, bins))

for bbox in bboxes2_resized:
    xi_min = max(0, int(bbox[0] * bins))
    yi_min = max(0, int(bbox[1] * bins))
    xi_max = min(bins, int(bbox[2] * bins))
    yi_max = min(bins, int(bbox[3] * bins))
    if xi_max > xi_min and yi_max > yi_min:
        # print(yi_min, yi_max, xi_min, xi_max)
        heatmap[yi_min:yi_max, xi_min:xi_max] += 1

heatmap_smooth = gaussian_filter(heatmap, sigma=3)
# heatmap_smooth = heatmap
# print(heatmap[0,0])
# print(heatmap[50,50])


# 基准 bbox
ref_bbox = [0,0,1,1]

# 可视化
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(
    heatmap_smooth,
    cmap="magma",
    extent=[0,1,0,1],
    origin="lower",
    alpha=0.9
)

# rect = plt.Rectangle(
#     (ref_bbox[0], ref_bbox[1]),
#     ref_bbox[2]-ref_bbox[0],
#     ref_bbox[3]-ref_bbox[1],
#     edgecolor="cyan",
#     facecolor="none",
#     lw=2,
#     linestyle="--"
# )
# ax.add_patch(rect)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Spatial Overlap Distribution")
# fig.colorbar(im, ax=ax, label="Frequency")
fig.colorbar(im, ax=ax, label="Frequency", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("/home/hang18/links/projects/rrg-vislearn/hang18/heatmap.png", dpi=300)