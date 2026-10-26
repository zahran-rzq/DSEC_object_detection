import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Urutan nama kelas persis sesuai tag <name> pada XML dataset Mendeley
CLASSES = ['bin', 'chair', 'door', 'fire extinguisher', 'human', 'shelf', 'table']

SRC_ROOT = Path("Indoor Objects.v1i.voc")  # sesuaikan dengan nama folder hasil ekstrak ZIP
DST_ROOT = Path("datasets")

for split in ["train", "valid", "test"]:
    src_dir = SRC_ROOT / split
    img_dst = DST_ROOT / split / "images"
    lbl_dst = DST_ROOT / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for xml_file in src_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in CLASSES:
                continue
            cls_id = CLASSES.index(cls_name)
            box = obj.find("bndbox")
            xmin = float(box.find("xmin").text)
            xmax = float(box.find("xmax").text)
            ymin = float(box.find("ymin").text)
            ymax = float(box.find("ymax").text)

            # Konversi ke format YOLO ternormalisasi (0.0 - 1.0)
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        # Simpan file label .txt dan salin gambar .jpg pasangannya
        (lbl_dst / f"{xml_file.stem}.txt").write_text("\n".join(yolo_lines))
        jpg_file = src_dir / f"{xml_file.stem}.jpg"
        if jpg_file.exists():
            shutil.copy2(jpg_file, img_dst / jpg_file.name)

# Buat file data.yaml sesuai Slide 30
yaml_content = f"""path: .
train: datasets/train/images
val: datasets/valid/images
test: datasets/test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
Path("data.yaml").write_text(yaml_content)
print("Konversi selesai! Folder datasets/ dan data.yaml siap digunakan.")