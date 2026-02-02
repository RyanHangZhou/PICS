import json
import os
import numpy as np
import cv2


def json_rectmask_to_local_mask(json_path, out_mask_path):
    """
    根据 json 里的 rectMask + polygon，生成局部小 mask：
    - 尺寸 = rectMask.height × rectMask.width
    - 坐标 = (x - xMin, y - yMin)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        objects = [data]
    else:
        objects = data

    assert len(objects) >= 1, "json 里没东西？"

    obj = objects[0]
    assert obj.get("contentType") == "polygon"

    pts = obj["content"]
    rect = obj["rectMask"]

    x_min = rect["xMin"]
    y_min = rect["yMin"]
    width = rect["width"]
    height = rect["height"]

    H = int(round(height))
    W = int(round(width))

    # 注意：这里就是“减 rectMask”
    poly = []
    for p in pts:
        x = p["x"] - x_min
        y = p["y"] - y_min
        poly.append([x, y])

    poly_np = np.array(poly, dtype=np.float32)
    poly_np = np.round(poly_np).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 255)

    cv2.imwrite(out_mask_path, mask)
    print("[mask saved]", out_mask_path)


def load_rect_from_json(json_path):
    """从 json 里把 rectMask 拿出来，返回 int 型 (x0, y0, W, H)"""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        obj = data
    else:
        obj = data[0]

    rect = obj["rectMask"]
    x_min = rect["xMin"]
    y_min = rect["yMin"]
    width = rect["width"]
    height = rect["height"]

    x0 = int(round(x_min))
    y0 = int(round(y_min))
    W  = int(round(width))
    H  = int(round(height))
    return x0, y0, W, H


if __name__ == "__main__":
    json_path   = "Rebuttal/R1_1/25.json"
    image_path  = "Rebuttal/R1_1/25.jpg"        # ⚠️ 现在这是整张图
    mask_path   = "Rebuttal/R1_1/mask_rect.png"
    out_crop    = "Rebuttal/R1_1/object_1_crop_masked_white.png"
    out_full    = "Rebuttal/R1_1/object_1_full_masked_white.png"

    # 1) 先根据 rectMask + polygon 生成“局部小 mask”（你说这一部是对的）
    json_rectmask_to_local_mask(json_path, mask_path)

    # 2) 读整图 & rect
    img_full = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x0, y0, W, H = load_rect_from_json(json_path)

    # 从整图裁出 rect 区域，对应 mask_rect.png
    crop = img_full[y0:y0+H, x0:x0+W]

    # 3) 读局部 mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print("full shape :", img_full.shape)
    print("crop shape :", crop.shape)
    print("mask shape :", mask.shape)

    # 保证 crop 和 mask 尺寸一致
    assert crop.shape[:2] == mask.shape[:2], "crop 和 mask_rect 尺寸没对上，检查一下 rectMask"

    # 4) 在 crop 上抠图：mask 内保留，外面白
    mask_bin = (mask > 128).astype(np.uint8)
    mask_3c  = np.repeat(mask_bin[:, :, None], 3, axis=2)

    white_bg_crop = np.ones_like(crop) * 255
    result_crop   = crop * mask_3c + white_bg_crop * (1 - mask_3c)

    cv2.imwrite(out_crop, result_crop)
    print("[saved crop result]", out_crop)

    # 5) 可选：把抠好的 crop 贴回整图，其它区域全白
    full_white = np.ones_like(img_full) * 255
    full_white[y0:y0+H, x0:x0+W] = result_crop
    cv2.imwrite(out_full, full_white)
    print("[saved full result]", out_full)

    # === 6) 生成局部 RGBD 图（D = mask，0/255） ===
    depth_crop = (mask_bin * 255).astype(np.uint8)          # [H, W]
    rgbd_crop  = np.dstack([crop, depth_crop])              # [H, W, 4]

    out_rgbd_crop = "Rebuttal/R1_1/object_1_crop_rgbd.png"
    cv2.imwrite(out_rgbd_crop, rgbd_crop)
    print("[saved crop RGBD]", out_rgbd_crop)

    # === 7) 生成整图 RGBD（可选）：RGB=原图，D=整图深度 ===
    # 整图深度图：先全 0，再把 rect 区域填成 depth_crop
    H_full, W_full = img_full.shape[:2]
    depth_full = np.zeros((H_full, W_full), dtype=np.uint8)
    depth_full[y0:y0+H, x0:x0+W] = depth_crop

    # 这里 RGB 用原图（不抠除白），你也可以换成 full_white 看需求
    rgbd_full = np.dstack([img_full, depth_full])

    out_rgbd_full = "Rebuttal/R1_1/object_1_full_rgbd.png"
    cv2.imwrite(out_rgbd_full, rgbd_full)
    print("[saved full RGBD]", out_rgbd_full)

