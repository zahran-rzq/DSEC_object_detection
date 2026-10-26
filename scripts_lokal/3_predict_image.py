# =============================================================================
#  Nama : Muhammad Zahran Razzaq
#  NRP  : 5024xxxxxx          <-- WAJIB diisi sebelum dikumpulkan (Nama_NRP.zip)
#  Mata Kuliah : Elektronika Cerdas — ETS Object Detection (Indoor)
#  File : scripts_lokal/3_predict_image.py
# =============================================================================
"""
LANGKAH 3 — Uji gambar statis dengan model hasil training (Slide 36).

Pola dosen:
    model = YOLO("MyYOLOModel.pt")
    results = model("foto.jpg")
    simpan / tampilkan hasil bounding box

Fitur tambahan:
  * Hitung jumlah objek per kelas (Object Counting, Slide 3).
  * --compare : bandingkan Pre-trained COCO (yolo12n.pt / yolo11n.pt) vs
    Custom Model (MyYOLOModel.pt) dalam satu gambar side-by-side.
  * Dukung satu file, satu folder, atau seluruh split valid/test.

Contoh:
    python scripts_lokal/3_predict_image.py --source contoh.jpg
    python scripts_lokal/3_predict_image.py --source datasets/test/images --limit 6
    python scripts_lokal/3_predict_image.py --source contoh.jpg --compare
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "hasil_prediksi"
EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def kumpulkan_gambar(source: str, limit: int | None) -> list[Path]:
    p = Path(source).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        gambar = sorted([f for f in p.iterdir() if f.suffix.lower() in EKSTENSI])
        if limit:
            gambar = gambar[:limit]
        return gambar
    raise SystemExit(f"[X] Sumber tidak ditemukan: {source}")


def hitung_objek(result, nama_kelas: dict) -> Counter:
    """Counter jumlah objek per kelas dari satu objek Results Ultralytics."""
    c = Counter()
    if result.boxes is None or len(result.boxes) == 0:
        return c
    for cid in result.boxes.cls.tolist():
        c[nama_kelas.get(int(cid), str(int(cid)))] += 1
    return c


def baca_model(path: Path):
    from ultralytics import YOLO
    if not path.exists():
        raise SystemExit(
            f"[X] Model '{path}' tidak ditemukan.\n"
            "    Jalankan dulu: python scripts_lokal/2_train_yolo.py"
        )
    return YOLO(str(path))


def prediksi_satu(model, gambar: Path, conf: float, imgsz: int, device, out_dir: Path,
                  suffix: str = "") -> tuple[Path, Counter, dict]:
    """Jalankan inferensi 1 gambar, simpan hasil beranotasi, kembalikan statistik."""
    results = model.predict(source=str(gambar), conf=conf, imgsz=imgsz,
                            device=device, verbose=False)
    r = results[0]
    nama = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    jumlah = hitung_objek(r, nama)

    out_dir.mkdir(parents=True, exist_ok=True)
    tujuan = out_dir / f"{gambar.stem}{suffix}_pred.jpg"
    r.save(filename=str(tujuan))     # simpan frame ber-anotasi bbox + label + conf

    return tujuan, jumlah, nama


def buat_perbandingan(gambar: Path, model_kustom, model_coco, conf: float, imgsz: int,
                      device, out_dir: Path) -> Path:
    """Side-by-side: Pre-trained COCO vs Custom Model (Slide 32/36 analisis)."""
    import cv2
    import numpy as np

    hasil = []
    for judul, model in [("Pre-trained COCO (kelas umum)", model_coco),
                         ("Custom Model MyYOLOModel.pt (7 kelas indoor)", model_kustom)]:
        r = model.predict(source=str(gambar), conf=conf, imgsz=imgsz,
                          device=device, verbose=False)[0]
        nama = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        jumlah = hitung_objek(r, nama)
        kanvas = r.plot()
        # banner judul
        cv2.rectangle(kanvas, (0, 0), (kanvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(kanvas, judul, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        ringkas = ", ".join(f"{k}:{v}" for k, v in sorted(jumlah.items())) or "tidak ada objek"
        cv2.putText(kanvas, ringkas[:70], (8, kanvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        hasil.append(kanvas)
        log(f"    {judul}: {dict(jumlah) if jumlah else 'tidak ada objek'}")

    tinggi = max(h.shape[0] for h in hasil)
    hasil = [cv2.resize(h, (int(h.shape[1] * tinggi / h.shape[0]), tinggi)) for h in hasil]
    gabung = np.hstack(hasil)

    out_dir.mkdir(parents=True, exist_ok=True)
    tujuan = out_dir / f"{gambar.stem}_compare_coco_vs_custom.jpg"
    cv2.imwrite(str(tujuan), gabung)
    return tujuan


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferensi gambar dengan MyYOLOModel.pt")
    parser.add_argument("--source", default="contoh.jpg",
                        help="file gambar / folder gambar")
    parser.add_argument("--model", default=str(ROOT_DIR / "MyYOLOModel.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah gambar folder")
    parser.add_argument("--compare", action="store_true",
                        help="bandingkan dengan pre-trained COCO (yolo12n.pt)")
    parser.add_argument("--coco", default="yolo12n.pt", help="model COCO pembanding")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    gambar_list = kumpulkan_gambar(args.source, args.limit)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    log("=" * 60)
    log(" INFERENSI GAMBAR — MyYOLOModel.pt")
    log("=" * 60)
    log(f" model  : {args.model}")
    log(f" sumber : {args.source} ({len(gambar_list)} gambar)")
    log(f" conf   : {args.conf} | imgsz {args.imgsz} | device {device}\n")

    model = baca_model(Path(args.model))

    model_coco = None
    if args.compare:
        # Model COCO boleh belum ada di disk -> Ultralytics mengunduh otomatis.
        from ultralytics import YOLO
        try:
            model_coco = baca_model(Path(args.coco))
        except SystemExit:
            log(f"[*] {args.coco} belum ada, mengunduh otomatis...")
            model_coco = YOLO(args.coco)

    total = Counter()
    for i, gambar in enumerate(gambar_list, start=1):
        tujuan, jumlah, _ = prediksi_satu(model, gambar, args.conf, args.imgsz,
                                          device, out_dir)
        total.update(jumlah)
        ringkas = ", ".join(f"{k}={v}" for k, v in sorted(jumlah.items())) or "tidak ada objek"
        log(f"  [{i}/{len(gambar_list)}] {gambar.name}: {ringkas}")
        log(f"           -> disimpan: {tujuan}")

        if model_coco is not None:
            banding = buat_perbandingan(gambar, model, model_coco, args.conf,
                                        args.imgsz, device, out_dir)
            log(f"           -> perbandingan: {banding}")

    log("\n=== TOTAL OBJEK TERDETEKSI (semua gambar) ===")
    if not total:
        log("  (tidak ada objek di atas threshold conf — coba turunkan --conf)")
    for k, v in total.most_common():
        log(f"  {k:<20}: {v}")
    log(f"\n[OK] Semua hasil ada di: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
