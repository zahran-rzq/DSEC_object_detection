# =============================================================================
#  Nama : Muhammad Zahran Razzaq
#  NRP  : 5024xxxxxx          <-- WAJIB diisi sebelum dikumpulkan (Nama_NRP.zip)
#  Mata Kuliah : Elektronika Cerdas — ETS Object Detection (Indoor)
#  File : scripts_lokal/2_train_yolo.py
# =============================================================================
"""
LANGKAH 2 — Transfer learning YOLO di dataset Indoor Objects (Slide 25 & 31).

Alur:
    model = YOLO("yolo12n.pt")                 # pre-trained COCO
    model.train(data="data.yaml", ...)         # transfer learning
    metrics = model.val()                      # Box(P), R, mAP50, mAP50-95
    simpan -> MyYOLOModel.pt                   # nama file wajib sesuai soal
    opsional -> model.export(format="onnx")

Hasil evaluasi otomatis dikumpulkan ke folder  ETS_results/ :
    results.png, confusion_matrix.png,
    val_batch0_labels.jpg, val_batch0_pred.jpg,
    training_curves.png, metrics_summary.txt, per_class_metrics.csv

Contoh pakai:
    python scripts_lokal/2_train_yolo.py                       # default yolo12n
    python scripts_lokal/2_train_yolo.py --model yolo11n.pt --epochs 50
    python scripts_lokal/2_train_yolo.py --device 0 --batch 16   # GPU
    python scripts_lokal/2_train_yolo.py --export-onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "ETS_results"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def pilih_device(arg: str) -> str | int | None:
    """device=0 bila GPU tersedia; 'cpu' bila tidak.

    Materi kuliah memakai device=0, tapi di laptop tanpa CUDA itu error,
    jadi kita deteksi otomatis.
    """
    if arg and arg.lower() != "auto":
        return int(arg) if arg.isdigit() else arg
    try:
        import torch
        if torch.cuda.is_available():
            log(f"[*] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            log("[*] Apple Silicon MPS terdeteksi.")
            return "mps"
    except Exception:
        pass
    log("[!] GPU tidak tersedia -> training di CPU (lebih lambat).")
    return "cpu"


def salin_hasil_jika_ada(save_dir: Path, hasil: Path) -> None:
    hasil.mkdir(parents=True, exist_ok=True)
    pola = [
        "results.png", "results.csv", "confusion_matrix.png",
        "confusion_matrix_normalized.png", "val_batch0_labels.jpg",
        "val_batch0_pred.jpg", "train_batch0.jpg", "labels.jpg",
        "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png", "BoxPR_curve.png",
    ]
    for nama in pola:
        src = save_dir / nama
        if src.exists():
            shutil.copy2(src, hasil / nama)
            log(f"    [copy] {nama}")


def plot_kurva_training(csv_path: Path, out_path: Path) -> None:
    """Plot box_loss, cls_loss, dfl_loss (train vs val) -> Slide 33."""
    if not csv_path.exists():
        log("[!] results.csv tidak ditemukan, plot kurva dilewati.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    pasangan = [
        ("train/box_loss", "val/box_loss", "box_loss"),
        ("train/cls_loss", "val/cls_loss", "cls_loss"),
        ("train/dfl_loss", "val/dfl_loss", "dfl_loss"),
    ]
    tersedia = [(t, v, n) for t, v, n in pasangan if t in df.columns and v in df.columns]
    if not tersedia:
        log("[!] Kolom loss tidak ditemukan di results.csv.")
        return

    fig, axes = plt.subplots(1, len(tersedia), figsize=(5 * len(tersedia), 4))
    if len(tersedia) == 1:
        axes = [axes]
    for ax, (kol_t, kol_v, judul) in zip(axes, tersedia):
        ax.plot(df["epoch"], df[kol_t], label="train", linewidth=1.8)
        ax.plot(df["epoch"], df[kol_v], label="val", linewidth=1.8, linestyle="--")
        ax.set_title(judul)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Kurva Loss Training vs Validasi (Slide 33)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"    [plot] {out_path.name}")


def plot_map_per_kelas(metrics, out_path: Path) -> None:
    """Grafik mAP50-95 per kelas — memenuhi permintaan 'metrik per kelas' (Slide 32)."""
    try:
        box = metrics.box
        indeks_kelas = list(box.ap_class_index)
        nama = [metrics.names[i] for i in indeks_kelas]
        ap50 = list(box.ap50)
        ap = list(box.ap)
    except Exception as exc:
        log(f"[!] Data per kelas tidak tersedia: {exc}")
        return
    if not nama:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(nama))
    ax.bar([i - 0.2 for i in x], ap50, width=0.4, label="mAP50")
    ax.bar([i + 0.2 for i in x], ap, width=0.4, label="mAP50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nama, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP")
    ax.set_title("mAP per Kelas — Custom Model (Slide 32)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (a50, a) in enumerate(zip(ap50, ap)):
        ax.text(i - 0.2, a50, f"{a50:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"    [plot] {out_path.name}")


def tulis_ringkasan_metrik(metrics, out_path: Path) -> None:
    """Simpan metrik ke teks + CSV (untuk laporan/analisis ETS)."""
    box = metrics.box
    baris = []
    baris.append("=" * 60)
    baris.append(" RINGKASAN EVALUASI — Custom Model (MyYOLOModel.pt)")
    baris.append("=" * 60)
    baris.append(f" Jumlah kelas  : {len(metrics.names)}")
    baris.append(f" Kelas         : {list(metrics.names.values())}")
    baris.append("")
    baris.append(f" Precision (Box P) : {box.mp:.4f}")
    baris.append(f" Recall    (Box R) : {box.mr:.4f}")
    baris.append(f" mAP50             : {box.map50:.4f}")
    baris.append(f" mAP50-95          : {box.map:.4f}")
    baris.append("")
    baris.append("-" * 60)
    baris.append(f"{'kelas':<20}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
    baris.append("-" * 60)
    for i, cid in enumerate(box.ap_class_index):
        nama = metrics.names[cid]
        p = box.p[i] if hasattr(box, "p") else float("nan")
        r = box.r[i] if hasattr(box, "r") else float("nan")
        baris.append(f"{nama:<20}{p:>10.3f}{r:>10.3f}{box.ap50[i]:>10.3f}{box.ap[i]:>12.3f}")
    teks = "\n".join(baris)
    out_path.write_text(teks, encoding="utf-8")
    log("\n" + teks + "\n")

    try:
        import pandas as pd
        df = pd.DataFrame({
            "class": [metrics.names[c] for c in box.ap_class_index],
            "precision": box.p,
            "recall": box.r,
            "mAP50": box.ap50,
            "mAP50-95": box.ap,
        })
        df.to_csv(out_path.with_name("per_class_metrics.csv"), index=False)
        log(f"    [csv] per_class_metrics.csv")
    except Exception as exc:
        log(f"[!] Gagal menyimpan CSV: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training YOLO untuk dataset Indoor Objects.")
    parser.add_argument("--data", default=str(ROOT_DIR / "data.yaml"), help="file data.yaml")
    parser.add_argument("--model", default="yolo12n.pt",
                        help="pre-trained: yolo12n.pt (Slide 28-31) / yolo11n.pt (Slide 25)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20, help="early stopping")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="0 / cpu / auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="ets_indoor")
    parser.add_argument("--nama-model", dest="nama_model", default="MyYOLOModel.pt",
                        help="nama file model hasil training")
    parser.add_argument("--export-onnx", action="store_true", help="ekspor ONNX (Slide 25)")
    parser.add_argument("--no-test", action="store_true", help="lewati evaluasi di test set")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(
            f"[X] {data_path} tidak ditemukan.\n"
            "    Jalankan dulu: python scripts_lokal/1_setup_dataset_mendeley.py"
        )

    from ultralytics import YOLO

    device = pilih_device(args.device)
    log("=" * 60)
    log(" TRAINING YOLO — ETS Object Detection (Indoor)")
    log("=" * 60)
    log(f" data.yaml : {data_path}")
    log(f" model     : {args.model}")
    log(f" epochs    : {args.epochs} (patience {args.patience}) | imgsz {args.imgsz} "
        f"| batch {args.batch} | device {device}")
    log("")

    model = YOLO(args.model)

    # ---------------- TRAINING ----------------
    hasil_train = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(ROOT_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,      # WAJIB: agar results.png, confusion_matrix.png, val_batch*.jpg dibuat
        seed=0,
        pretrained=True,
        optimizer="auto",
        cache=False,
    )
    save_dir = Path(getattr(hasil_train, "save_dir", ROOT_DIR / "runs" / args.name))
    log(f"\n[OK] Training selesai. Artefak Ultralytics: {save_dir}")

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"

    # ---------------- SIMPAN MODEL FINAL ----------------
    model_final = ROOT_DIR / args.nama_model
    if best_pt.exists():
        shutil.copy2(best_pt, model_final)
        log(f"[OK] Model terlatih disimpan sebagai: {model_final}  (dari {best_pt.name})")

    # ---------------- VALIDASI + METRIK ----------------
    model_terbaik = YOLO(str(model_final if model_final.exists() else best_pt))

    log("\n[*] Evaluasi di VALIDATION set...")
    metrics_val = model_terbaik.val(data=str(data_path), split="val",
                                    imgsz=args.imgsz, batch=args.batch, device=device,
                                    plots=True, project=str(ROOT_DIR / "runs"),
                                    name=f"{args.name}_val", exist_ok=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tulis_ringkasan_metrik(metrics_val, OUT_DIR / "metrics_summary.txt")
    plot_map_per_kelas(metrics_val, OUT_DIR / "map_per_class.png")

    if not args.no_test:
        log("[*] Evaluasi di TEST set...")
        metrics_test = model_terbaik.val(data=str(data_path), split="test",
                                         imgsz=args.imgsz, batch=args.batch, device=device,
                                         plots=True, project=str(ROOT_DIR / "runs"),
                                         name=f"{args.name}_test", exist_ok=True)
        log(f"  TEST -> Precision {metrics_test.box.mp:.4f} | Recall {metrics_test.box.mr:.4f} "
            f"| mAP50 {metrics_test.box.map50:.4f} | mAP50-95 {metrics_test.box.map:.4f}")

    # ---------------- KUMPULKAN ARTEFAK VISUAL (Slide 33-35) ----------------
    log("\n[*] Mengumpulkan artefak evaluasi ke ETS_results/ ...")
    salin_hasil_jika_ada(save_dir, OUT_DIR)
    for sub in [f"{args.name}_val", f"{args.name}_test"]:
        salin_hasil_jika_ada(ROOT_DIR / "runs" / sub, OUT_DIR)
    plot_kurva_training(OUT_DIR / "results.csv", OUT_DIR / "training_curves.png")

    # ---------------- EXPORT ONNX (opsional, Slide 25) ----------------
    if args.export_onnx:
        log("\n[*] Ekspor ONNX...")
        try:
            jalur = YOLO(str(model_final)).export(format="onnx", imgsz=args.imgsz, opset=12)
            log(f"[OK] ONNX: {jalur}")
            shutil.copy2(jalur, ROOT_DIR / "MyYOLOModel.onnx")
        except Exception as exc:
            log(f"[!] Ekspor ONNX gagal: {exc}")

    log("\n" + "=" * 60)
    log(" SELESAI. Langkah berikutnya:")
    log("   python scripts_lokal/3_predict_image.py --source foto.jpg")
    log("   python scripts_lokal/4_indoor_security_webcam.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
