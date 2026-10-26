# =============================================================================
#  Nama : Muhammad Zahran Razzaq
#  NRP  : 5024xxxxxx          <-- WAJIB diisi sebelum dikumpulkan (Nama_NRP.zip)
#  Mata Kuliah : Elektronika Cerdas — ETS Object Detection (Indoor)
# =============================================================================
"""
voc_to_yolo.py  —  Konverter anotasi Pascal VOC (.xml) -> YOLO (.txt)
                   untuk dataset "Indoor Object Detection" (Mendeley / Roboflow).

Versi ringkas / standalone. Jika butuh pipeline lengkap
(download -> auto-unzip bertumpuk -> subset -> class distribution -> data.yaml),
gunakan:  python scripts_lokal/1_setup_dataset_mendeley.py

Kenapa file ini ada?
  Dataset Mendeley diekspor dari Roboflow dalam format **Pascal VOC**:
  tiap gambar .jpg punya pasangan .xml dengan tag <bndbox><xmin><ymin><xmax><ymax>.
  Ultralytics YOLO hanya menerima label .txt ternormalisasi:
        class_id  x_center  y_center  width  height      (semua 0.0 .. 1.0)

Cara pakai (dari root repo, setelah ZIP dataset diekstrak):
    python voc_to_yolo.py
    python voc_to_yolo.py --src "Indoor Objects.v1i.voc" --dst datasets
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Konfigurasi
# -----------------------------------------------------------------------------
# Urutan & ejaan WAJIB persis seperti tag <name> di XML dataset.
# Perhatikan: 'fire extinguisher' memakai SPASI, bukan tanda hubung!
CLASSES = ["bin", "chair", "door", "fire extinguisher", "human", "shelf", "table"]

SPLITS = ["train", "valid", "test"]

# Pemetaan nama folder sumber -> nama split di data.yaml (Ultralytics)
SPLIT_ALIAS = {"train": "train", "valid": "val", "test": "test"}


def tautkan_atau_salin(src: Path, dst: Path) -> None:
    """Hardlink gambar (instan, hemat disk) dengan fallback copy.

    Di Google Colab, hardlink hanya bekerja bila src & dst satu filesystem
    (biasanya /content). Kalau mount Google Drive -> otomatis pindah ke copy.
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)          # cepat: hanya menambah entry direktori
    except OSError:
        shutil.copy2(src, dst)     # fallback aman lintas filesystem


def baca_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Ambil (width, height, [(nama_kelas, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Tag <size> tidak ditemukan di {xml_path.name}")
    lebar = int(float(size.findtext("width", "0")))
    tinggi = int(float(size.findtext("height", "0")))
    if lebar <= 0 or tinggi <= 0:
        raise ValueError(f"<size> tidak valid ({lebar}x{tinggi}) di {xml_path.name}")

    objek = []
    for obj in root.findall("object"):
        nama = (obj.findtext("name") or "").strip()
        if not nama:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        # Jaga-jaga bila koordinat tertukar di anotasi asli
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        objek.append((nama, xmin, ymin, xmax, ymax))
    return lebar, tinggi, objek


def ke_baris_yolo(nama_kelas: str, xmin: float, ymin: float, xmax: float, ymax: float,
                  lebar: int, tinggi: int, kelas_aktif: list[str]) -> str | None:
    """Ubah satu bbox VOC -> baris label YOLO ternormalisasi (di-clamp 0..1)."""
    if nama_kelas not in kelas_aktif:
        return None
    cls_id = kelas_aktif.index(nama_kelas)

    # Clamp ke batas gambar: sebagian bbox Roboflow keluar sedikit dari frame
    # -> bila tidak di-clamp, Ultralytics mengeluarkan warning "out of bounds".
    xmin = min(max(xmin, 0.0), float(lebar))
    xmax = min(max(xmax, 0.0), float(lebar))
    ymin = min(max(ymin, 0.0), float(tinggi))
    ymax = min(max(ymax, 0.0), float(tinggi))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 1.0 or bh <= 1.0:      # bbox degenerat / hasil clamp habis -> buang
        return None

    x_center = (xmin + xmax) / 2.0 / lebar
    y_center = (ymin + ymax) / 2.0 / tinggi
    bw /= lebar
    bh /= tinggi

    # Clamp terakhir untuk keamanan numerik
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    return f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"


def konversi(src_root: Path, dst_root: Path,
             kelas_aktif: list[str] | None = None) -> dict:
    """Konversi seluruh split. Mengembalikan statistik untuk laporan ETS."""
    kelas_aktif = kelas_aktif or CLASSES
    stat = {"gambar": Counter(), "bbox": Counter(), "kelas": Counter(), "gagal": []}

    for split in SPLITS:
        src_dir = src_root / split
        if not src_dir.is_dir():
            print(f"[!] Folder '{src_dir}' tidak ada — split '{split}' dilewati.")
            continue

        img_dst = dst_root / split / "images"
        lbl_dst = dst_root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for xml_path in sorted(src_dir.glob("*.xml")):
            try:
                lebar, tinggi, objek = baca_xml(xml_path)
            except Exception as exc:                      # XML rusak -> catat & lanjut
                stat["gagal"].append((xml_path.name, str(exc)))
                continue

            baris = []
            for nama, xmin, ymin, xmax, ymax in objek:
                b = ke_baris_yolo(nama, xmin, ymin, xmax, ymax,
                                  lebar, tinggi, kelas_aktif)
                if b:
                    baris.append(b)
                    stat["kelas"][nama] += 1

            # Gambar tanpa objek tetap disimpan sebagai "background image"
            # (label kosong) — membantu menekan false positive.
            (lbl_dst / f"{xml_path.stem}.txt").write_text("\n".join(baris), encoding="utf-8")

            jpg = src_dir / f"{xml_path.stem}.jpg"
            if jpg.exists():
                tautkan_atau_salin(jpg, img_dst / jpg.name)
                stat["gambar"][split] += 1
                stat["bbox"][split] += len(baris)
            else:
                stat["gagal"].append((xml_path.name, "pasangan .jpg tidak ditemukan"))

    return stat


def tulis_data_yaml(dst_root: Path, yaml_path: Path, kelas_aktif: list[str]) -> None:
    """Tulis data.yaml dengan path ABSOLUT.

    PENTING (jebakan Colab): jangan pakai `path: .` atau path relatif.
    Ultralytics akan menggabungkannya dengan lokasi file / working dir dan
    sering berakhir mencari ke '/content/datasets/datasets/...'.
    """
    konten = {
        "path": dst_root.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(kelas_aktif),
        "names": kelas_aktif,
    }
    yaml_path.write_text(
        yaml.safe_dump(konten, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"\n[OK] {yaml_path} ditulis (path absolut):")
    print(yaml_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konversi anotasi Pascal VOC -> YOLO untuk dataset Indoor Objects."
    )
    parser.add_argument("--src", default="Indoor Objects.v1i.voc",
                        help="folder hasil ekstrak berisi train/ valid/ test/")
    parser.add_argument("--dst", default="datasets",
                        help="folder tujuan berstruktur datasets/<split>/{images,labels}")
    parser.add_argument("--yaml", default="data.yaml", help="lokasi file data.yaml")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="subset kelas, mis. --classes chair door")
    args = parser.parse_args()

    src_root = Path(args.src).expanduser()
    dst_root = Path(args.dst).expanduser()

    if not src_root.is_dir():
        raise SystemExit(
            f"[X] Folder sumber '{src_root}' tidak ditemukan.\n"
            "    Ekstrak dulu 'Indoor Objects.v1i.voc.zip' di folder ini."
        )

    kelas_aktif = args.classes or CLASSES
    tak_dikenal = [k for k in kelas_aktif if k not in CLASSES]
    if tak_dikenal:
        raise SystemExit(f"[X] Kelas tidak dikenal: {tak_dikenal}\n    Pilihan: {CLASSES}")

    print(f"[*] Konversi VOC -> YOLO\n    src = {src_root.resolve()}\n    dst = {dst_root.resolve()}")
    stat = konversi(src_root, dst_root, kelas_aktif)
    tulis_data_yaml(dst_root, Path(args.yaml), kelas_aktif)

    print("\n=== RINGKASAN KONVERSI ===")
    for split in SPLITS:
        print(f"  {split:<6}: {stat['gambar'][split]:>6} gambar | "
              f"{stat['bbox'][split]:>6} bbox")
    print("\n  Distribusi per kelas (semua split):")
    for nama in kelas_aktif:
        print(f"    - {nama:<18}: {stat['kelas'][nama]:>6}")
    if stat["gagal"]:
        print(f"\n  [!] {len(stat['gagal'])} file bermasalah, contoh: {stat['gagal'][:3]}")
    print("\nSelesai. Lanjut:  python scripts_lokal/2_train_yolo.py")


if __name__ == "__main__":
    main()
