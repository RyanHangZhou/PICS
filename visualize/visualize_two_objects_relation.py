import os
from pathlib import Path
from PIL import Image
from math import ceil

def resize_keep_ratio(img, target_height):
    """Resize image to target height while keeping aspect ratio."""
    w, h = img.size
    ratio = target_height / h
    return img.resize((int(w * ratio), target_height), Image.ANTIALIAS)

def concat_images_grid(images, N, M, bg_color=(255, 255, 255)):
    """Concatenate N x M images into a grid, filling blanks if needed."""
    max_height = max(img.height for img in images)
    resized_imgs = [resize_keep_ratio(img, max_height) for img in images]

    # Pad to fill full N x M grid
    padded_imgs = resized_imgs + [Image.new("RGB", (1, max_height), bg_color)] * (N * M - len(resized_imgs))

    # Reshape into grid
    grid = [padded_imgs[i * M:(i + 1) * M] for i in range(N)]

    # Compute max width for each column
    col_widths = [max(row[j].width for row in grid) for j in range(M)]
    total_width = sum(col_widths)
    total_height = max_height * N

    canvas = Image.new("RGB", (total_width, total_height), bg_color)

    y_offset = 0
    for row in grid:
        x_offset = 0
        for j, img in enumerate(row):
            canvas.paste(img, (x_offset, y_offset))
            x_offset += col_widths[j]
        y_offset += max_height

    return canvas

def collect_image_paths_from_subfolders(root_dir):
    """Only collect paths, not actual images."""
    return sorted(Path(root_dir).glob("*/highlighted_image.png"))

def main(input_root, output_dir, N, M):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths_from_subfolders(input_root)
    total = len(image_paths)
    if total == 0:
        print("No images found.")
        return

    per_page = N * M
    num_pages = ceil(total / per_page)
    print(f"Found {total} images, generating {num_pages} pages.")

    for i in range(num_pages):
        start = i * per_page
        end = min(start + per_page, total)
        page_paths = image_paths[start:end]

        page_images = []
        for path in page_paths:
            try:
                with Image.open(path) as img:
                    page_images.append(img.copy())  # load into memory only this batch
            except Exception as e:
                print(f"Failed to open {path}: {e}")

        grid_img = concat_images_grid(page_images, N, M)
        out_path = output_dir / f"grid_{i+1:03d}.jpg"
        grid_img.save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Visualize data")
    parser.add_argument("--input_root", type=str, required=True, help="Path to the constructed dataset.")
    parser.add_argument("--output_dir", type=str, default='bin', help="Path to the concatenation directory.")
    args = parser.parse_args()

    N = 4  # row number
    M = 5  # column number
    main(args.input_root, args.output_dir, N, M)
