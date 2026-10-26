# =============================================================================
#  Nama : Muhammad Zahran Razzaq
#  NRP  : 5024xxxxxx          <-- WAJIB diisi sebelum dikumpulkan (Nama_NRP.zip)
#  Mata Kuliah : Elektronika Cerdas — ETS Object Detection (Indoor)
#  File : scripts_lokal/1_setup_dataset_mendeley.py
# =============================================================================
"""
LANGKAH 1 — Siapkan dataset Indoor Objects (Mendeley) menjadi format YOLO.

Yang dikerjakan script ini (sekali jalan):
  1. Unduh ZIP dataset dari Mendeley (langsung, tanpa lapisan luar) — atau pakai
     ZIP lokal / hasil "Download All" yang masih membungkus ZIP di dalam ZIP.
  2. Ekstrak otomatis, termasuk menangani NESTED ZIP
     ("Indoor Object Detection Dataset.zip" -> "Indoor Objects.v1i.voc.zip").
  3. Konversi anotasi Pascal VOC (.xml) -> YOLO (.txt) ternormalisasi + clamp 0..1.
     Gambar dipindahkan dengan os.link (hardlink, instan) + fallback shutil.copy2.
  4. Susun struktur wajib Slide 30: datasets/<split>/{images,labels}.
  5. Buat data.yaml dengan path ABSOLUT (menghindari bug /content/datasets/datasets).
  6. Simpan grafik Class Distribution (Slide 31) -> class_distribution.png.

Jalankan dari ROOT repo:
    python scripts_lokal/1_setup_dataset_mendeley.py
    python scripts_lokal/1_setup_dataset_mendeley.py --zip "Indoor Objects.v1i.voc.zip"
    python scripts_lokal/1_setup_dataset_mendeley.py --max-train 2000 --classes chair door
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

# -----------------------------------------------------------------------------
# KONFIGURASI — ubah di sini bila perlu
# -----------------------------------------------------------------------------
# Link unduh langsung file ZIP "Indoor Objects.v1i.voc" (433 MB, tanpa lapisan luar)
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/3ggxwf2vpr/files/"
    "e9e913c0-e5eb-476b-a296-2ab55e55ada5/file_downloaded"
)

ROOT_DIR = Path(__file__).resolve().parent.parent   # root repo
RAW_DIR = ROOT_DIR / "_raw"                          # tempat ZIP & hasil ekstrak
DATASET_DIR = ROOT_DIR / "datasets"                  # output YOLO
DATA_YAML = ROOT_DIR / "data.yaml"
PLOT_PATH = ROOT_DIR / "class_distribution.png"

# 7 kelas persis sesuai tag <name> di XML ('fire extinguisher' pakai SPASI!)
CLASSES = ["bin", "chair", "door", "fire extinguisher", "human", "shelf", "table"]

SPLITS = ["train", "valid", "test"]

# --- Panel kontrol soal ETS -------------------------------------------------
SELECTED_CLASSES: list[str] | None = None   # None = 7 kelas; atau ['chair','door']
MAX_TRAIN_IMAGES: int | None = None         # None = semua (~12.012); mis. 2000
MAX_VALID_IMAGES: int | None = None         # None = semua (490)
MAX_TEST_IMAGES: int | None = None          # None = semua (245)

# True  = buang split Roboflow (94/4/2) lalu split ulang 70/15/15 (Slide 21).
# False = pakai split asli dataset (default: paling aman & paling cepat).
RESPLIT_70_15_15 = False
SEED = 42

# -----------------------------------------------------------------------------
# Utilitas
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ukuran_manusia(n: float) -> str:
    for satuan in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {satuan}"
        n /= 1024
    return f"{n:.1f} TB"


def unduh(url: str, tujuan: Path) -> Path:
    """Unduh dengan progress bar sederhana (requests -> fallback urllib)."""
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    if tujuan.exists() and tujuan.stat().st_size > 1_000_000:
        log(f"[=] ZIP sudah ada, lewati unduh: {tujuan} ({ukuran_manusia(tujuan.stat().st_size)})")
        return tujuan

    log(f"[*] Mengunduh dataset dari Mendeley...\n    {url}")
    t0 = time.time()
    try:
        import requests
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(tujuan, "wb") as f:
                terunduh = 0
                for potong in r.iter_content(chunk_size=1024 * 1024):
                    f.write(potong)
                    terunduh += len(potong)
                    if total:
                        persen = terunduh / total * 100
                        bar = "#" * int(persen // 2)
                        print(f"\r    [{bar:<50}] {persen:5.1f}%  "
                              f"{ukuran_manusia(terunduh)}/{ukuran_manusia(total)}",
                              end="", flush=True)
        print()
    except Exception as exc:
        log(f"[!] Gagal lewat requests ({exc}); mencoba urllib...")
        import urllib.request

        def _hook(blok: int, ukuran_blok: int, total: int) -> None:
            if total > 0:
                persen = min(blok * ukuran_blok / total * 100, 100)
                print(f"\r    {persen:5.1f}%  ({ukuran_manusia(blok * ukuran_blok)})",
                      end="", flush=True)

        urllib.request.urlretrieve(url, tujuan, reporthook=_hook)
        print()

    log(f"[OK] Selesai dalam {time.time() - t0:.1f} s -> {tujuan} "
        f"({ukuran_manusia(tujuan.stat().st_size)})")
    return tujuan


def ekstrak_bertumpuk(zip_path: Path, tujuan: Path, maks_level: int = 3) -> None:
    """Ekstrak ZIP, lalu ekstrak lagi ZIP yang ada di dalamnya (nested ZIP).

    Kasus nyata: tombol "Download All" Mendeley menghasilkan
    "Indoor Object Detection Dataset.zip" yang isinya masih
    "Indoor Objects.v1i.voc.zip".
    """
    tujuan.mkdir(parents=True, exist_ok=True)
    antre = [zip_path]
    sudah = set()

    for level in range(1, maks_level + 1):
        if not antre:
            break
        log(f"[*] Ekstrak level {level}: {len(antre)} arsip...")
        berikutnya = []
        for z in antre:
            if z in sudah:
                continue
            sudah.add(z)
            log(f"    - {z.name} ({ukuran_manusia(z.stat().st_size)})")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tujuan)
                for nama in zf.namelist():
                    if nama.lower().endswith(".zip"):
                        kandidat = tujuan / nama
                        if kandidat.exists():
                            berikutnya.append(kandidat)
        antre = berikutnya

    if antre:
        log(f"[!] Masih ada ZIP belum diekstrak: {[z.name for z in antre]}")


def cari_root_dataset(folder: Path, kedalaman_maks: int = 3) -> Path:
    """Cari folder yang langsung berisi subfolder train/ dan valid/."""
    kandidat = []
    for path in [folder, *folder.rglob("*")]:
        if not path.is_dir():
            continue
        if len(path.relative_to(folder).parts) > kedalaman_maks:
            continue
        if (path / "train").is_dir() and (path / "valid").is_dir():
            kandidat.append(path)
    if not kandidat:
        raise SystemExit(
            f"[X] Tidak menemukan folder berisi 'train/' dan 'valid/' di {folder}.\n"
            "    Periksa isi folder ekstrak, atau arahkan --zip ke file yang benar."
        )
    kandidat.sort(key=lambda p: len(p.parts))
    root = kandidat[0]
    log(f"[OK] Root dataset terdeteksi: {root}")
    return root


# -----------------------------------------------------------------------------
# Konversi VOC -> YOLO
# -----------------------------------------------------------------------------
def baca_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("tag <size> tidak ada")
    lebar = int(float(size.findtext("width", "0")))
    tinggi = int(float(size.findtext("height", "0")))
    if lebar <= 0 or tinggi <= 0:
        raise ValueError(f"<size> tidak valid ({lebar}x{tinggi})")

    objek = []
    for obj in root.findall("object"):
        nama = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not nama or box is None:
            continue
        try:
            xmin = float(box.findtext("xmin")); ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax")); ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        objek.append((nama, xmin, ymin, xmax, ymax))
    return lebar, tinggi, objek


def tautkan(src: Path, dst: Path) -> None:
    """os.link (hardlink) + fallback shutil.copy2."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def kumpulkan_pasangan(root: Path) -> list[tuple[Path, Path]]:
    """Kumpulkan semua pasangan (gambar .jpg, anotasi .xml) dari semua split."""
    pasangan = []
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            continue
        for xml in sorted(d.glob("*.xml")):
            jpg = d / f"{xml.stem}.jpg"
            if jpg.exists():
                pasangan.append((jpg, xml))
    return pasangan


def pilih(pasangan: list[tuple[Path, Path]], maks: int | None, seed: int) -> list[tuple[Path, Path]]:
    """Subset deterministik (seed tetap) — aman diulang saat uji coba."""
    if maks is None or maks >= len(pasangan):
        return pasangan
    return random.Random(seed).sample(pasangan, maks)


def konversi(root: Path, kelas_aktif: list[str], resplit: bool,
             max_train: int | None, max_valid: int | None, max_test: int | None,
             seed: int) -> dict:
    dst = DATASET_DIR
    stat = {
        "gambar": Counter(), "bbox": Counter(), "kelas": Counter(),
        "gambar_per_kelas": Counter(), "kosong": Counter(), "gagal": [],
        "kelas_asing": Counter(),
        "kelas_split": {},        # {split: Counter(nama kelas -> jumlah bbox)}
        "gambar_split_kelas": {}, # {split: Counter(nama kelas -> jumlah gambar)}
    }

    # --- 1) tentukan daftar pasangan per split tujuan ---
    if resplit:
        semua = kumpulkan_pasangan(root)
        random.Random(seed).shuffle(semua)
        n = len(semua)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        daftar = {
            "train": semua[:n_train],
            "valid": semua[n_train:n_train + n_val],
            "test": semua[n_train + n_val:],
        }
        log(f"[*] Re-split 70/15/15 dari total {n} gambar: "
            f"{len(daftar['train'])}/{len(daftar['valid'])}/{len(daftar['test'])}")
    else:
        daftar = {}
        for split in SPLITS:
            d = root / split
            ps = []
            if d.is_dir():
                for xml in sorted(d.glob("*.xml")):
                    jpg = d / f"{xml.stem}.jpg"
                    if jpg.exists():
                        ps.append((jpg, xml))
            daftar[split] = ps
        log("[*] Memakai split asli dataset Roboflow (train/valid/test).")

    daftar["train"] = pilih(daftar["train"], max_train, seed)
    daftar["valid"] = pilih(daftar["valid"], max_valid, seed + 1)
    daftar["test"] = pilih(daftar["test"], max_test, seed + 2)

    # --- 2) konversi per split ---
    for split in SPLITS:
        img_dst = dst / split / "images"
        lbl_dst = dst / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        stat["kelas_split"][split] = Counter()
        stat["gambar_split_kelas"][split] = Counter()

        for i, (jpg, xml) in enumerate(daftar[split], start=1):
            try:
                lebar, tinggi, objek = baca_xml(xml)
            except Exception as exc:
                stat["gagal"].append((xml.name, str(exc)))
                continue

            baris = []
            kelas_di_gambar = set()
            for nama, xmin, ymin, xmax, ymax in objek:
                if nama not in CLASSES:
                    stat["kelas_asing"][nama] += 1
                    continue
                if nama not in kelas_aktif:
                    continue
                cid = kelas_aktif.index(nama)

                # clamp ke batas gambar (cegah warning out-of-bounds Ultralytics)
                xmin = min(max(xmin, 0.0), float(lebar))
                xmax = min(max(xmax, 0.0), float(lebar))
                ymin = min(max(ymin, 0.0), float(tinggi))
                ymax = min(max(ymax, 0.0), float(tinggi))
                bw, bh = xmax - xmin, ymax - ymin
                if bw <= 1.0 or bh <= 1.0:
                    continue

                xc = ((xmin + xmax) / 2.0) / lebar
                yc = ((ymin + ymax) / 2.0) / tinggi
                bw, bh = bw / lebar, bh / tinggi
                baris.append(f"{cid} {min(max(xc, 0), 1):.6f} {min(max(yc, 0), 1):.6f} "
                             f"{min(bw, 1):.6f} {min(bh, 1):.6f}")
                stat["kelas"][nama] += 1
                stat["kelas_split"][split][nama] += 1
                kelas_di_gambar.add(nama)

            if not baris:
                # SELECTED_CLASSES dipakai: buang gambar yang tidak punya kelas target
                if kelas_aktif != CLASSES:
                    continue
                stat["kosong"][split] += 1      # background image: label kosong

            (lbl_dst / f"{xml.stem}.txt").write_text("\n".join(baris), encoding="utf-8")
            tautkan(jpg, img_dst / jpg.name)
            stat["gambar"][split] += 1
            stat["bbox"][split] += len(baris)
            for k in kelas_di_gambar:
                stat["gambar_per_kelas"][k] += 1
                stat["gambar_split_kelas"][split][k] += 1

            if i % 2000 == 0:
                log(f"    {split}: {i} gambar diproses...")

    return stat


# -----------------------------------------------------------------------------
# data.yaml + visualisasi
# -----------------------------------------------------------------------------
def tulis_yaml(kelas_aktif: list[str]) -> None:
    import yaml

    konten = {
        # path ABSOLUT — jangan relatif, ini yang bikin Colab cari
        # ke /content/datasets/datasets/... dan gagal.
        "path": DATASET_DIR.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(kelas_aktif),
        "names": kelas_aktif,
    }
    DATA_YAML.write_text(yaml.safe_dump(konten, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
    log(f"\n[OK] data.yaml ditulis di {DATA_YAML}\n" + "-" * 56)
    log(DATA_YAML.read_text(encoding="utf-8").rstrip())
    log("-" * 56)


def plot_distribusi(stat: dict, kelas_aktif: list[str]) -> None:
    """Class distribution bar chart (Slide 31): total + per split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = [stat["kelas"][k] for k in kelas_aktif]
    gambar = [stat["gambar_per_kelas"][k] for k in kelas_aktif]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    batang = ax.bar(kelas_aktif, total, color="#2b7bba", edgecolor="black", linewidth=0.6)
    ax.set_title("Class Distribution — jumlah BBox per kelas (semua split)")
    ax.set_ylabel("Jumlah instance (bbox)")
    ax.tick_params(axis="x", rotation=25)
    for b, v in zip(batang, total):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    lebar = 0.26
    for i, split in enumerate(SPLITS):
        nilai = [stat["gambar_split_kelas"][split].get(k, 0) for k in kelas_aktif]
        ax.bar([x + (i - 1) * lebar for x in range(len(kelas_aktif))], nilai,
               width=lebar, label=split)
    ax.set_xticks(range(len(kelas_aktif)))
    ax.set_xticklabels(kelas_aktif, rotation=25)
    ax.set_title("Jumlah gambar yang memuat tiap kelas")
    ax.set_ylabel("Jumlah gambar")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Indoor Object Detection — Class Distribution (Slide 31)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Grafik class distribution disimpan: {PLOT_PATH}")

    # Tabel teks distribusi per split (untuk laporan)
    log("\n=== DISTRIBUSI PER SPLIT (jumlah bbox per kelas) ===")
    header = f"{'kelas':<20}" + "".join(f"{s:>10}" for s in SPLITS)
    log(header)
    log("-" * len(header))
    for k in kelas_aktif:
        jml = [stat["kelas_split"].get(s, Counter()).get(k, 0) for s in SPLITS]
        log(f"{k:<20}" + "".join(f"{v:>10,}" for v in jml))


def validasi(kelas_aktif: list[str]) -> None:
    """Cek cepat: jumlah gambar vs label, dan label di luar range."""
    log("\n=== VALIDASI HASIL KONVERSI ===")
    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_img = sum(1 for _ in img_dir.glob("*.jpg")) if img_dir.is_dir() else 0
        n_lbl = sum(1 for _ in lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else 0
        status = "OK" if n_img == n_lbl and n_img > 0 else "PERIKSA!"
        log(f"  {split:<6}: {n_img:>6} gambar | {n_lbl:>6} label  -> {status}")

    salah = 0
    for lbl in DATASET_DIR.rglob("labels/*.txt"):
        for baris in lbl.read_text(encoding="utf-8").splitlines():
            bagian = baris.split()
            if len(bagian) != 5:
                salah += 1
                continue
            try:
                cid = int(bagian[0])
                nilai = [float(x) for x in bagian[1:]]
            except ValueError:
                salah += 1
                continue
            if cid < 0 or cid >= len(kelas_aktif) or any(v < -1e-6 or v > 1 + 1e-6 for v in nilai):
                salah += 1
    log(f"  Baris label tidak valid / out-of-bounds: {salah}")
    if salah == 0:
        log("  -> Semua label bersih, siap dipakai Ultralytics tanpa warning.")
    log(f"\n  nc = {len(kelas_aktif)} | names = {kelas_aktif}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan dataset Indoor Objects -> YOLO.")
    parser.add_argument("--zip", dest="zip_path", default=None,
                        help="path ZIP dataset lokal (default: cari di _raw/ atau root)")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="URL unduh langsung")
    parser.add_argument("--no-download", action="store_true",
                        help="jangan unduh, pakai ZIP/folder lokal saja")
    parser.add_argument("--classes", nargs="*", default=SELECTED_CLASSES,
                        help="subset kelas, mis. --classes chair door")
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_IMAGES)
    parser.add_argument("--max-valid", type=int, default=MAX_VALID_IMAGES)
    parser.add_argument("--max-test", type=int, default=MAX_TEST_IMAGES)
    parser.add_argument("--resplit", action="store_true", default=RESPLIT_70_15_15,
                        help="split ulang 70/15/15 (abaikan split asli Roboflow)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    log("=" * 60)
    log(" SETUP DATASET — Indoor Object Detection (Mendeley, CC BY 4.0)")
    log("=" * 60)

    kelas_aktif = args.classes or CLASSES
    tak_dikenal = [k for k in kelas_aktif if k not in CLASSES]
    if tak_dikenal:
        raise SystemExit(f"[X] Kelas tidak dikenal: {tak_dikenal}\n    Pilihan: {CLASSES}")
    if kelas_aktif != CLASSES:
        log(f"[*] Mode SUBSET KELAS: {kelas_aktif} (nc={len(kelas_aktif)})")

    # ---------- tahap A: siapkan file ZIP ----------
    zip_path = Path(args.zip_path).expanduser() if args.zip_path else None
    if zip_path is None:
        for kandidat in [RAW_DIR / "Indoor Objects.v1i.voc.zip",
                         ROOT_DIR / "Indoor Objects.v1i.voc.zip",
                         ROOT_DIR / "Indoor Object Detection Dataset.zip"]:
            if kandidat.exists():
                zip_path = kandidat
                break

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- tahap B: cari folder dataset yang sudah siap ----------
    root_dataset = None
    for calon in [RAW_DIR / "Indoor Objects.v1i.voc", ROOT_DIR / "Indoor Objects.v1i.voc"]:
        if (calon / "train").is_dir():
            root_dataset = calon
            log(f"[=] Folder dataset sudah diekstrak: {calon}")
            break

    if root_dataset is None:
        if zip_path is not None:
            if not zip_path.is_absolute():
                zip_path = (ROOT_DIR / zip_path).resolve()
            log(f"[*] Menggunakan ZIP lokal: {zip_path}")
        elif args.no_download:
            raise SystemExit("[X] --no-download dipakai tapi tidak ada ZIP/folder dataset.")
        else:
            zip_path = RAW_DIR / "Indoor Objects.v1i.voc.zip"
            unduh(args.url, zip_path)

        ekstrak_bertumpuk(zip_path, RAW_DIR)
        root_dataset = cari_root_dataset(RAW_DIR)
    else:
        log("[=] Ekstraksi dilewati.")

    # ---------- tahap C: konversi ----------
    log("\n[*] Konversi VOC (.xml) -> YOLO (.txt)...")
    if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.txt")):
        cadangan = DATASET_DIR.with_name(f"datasets_lama_{int(time.time())}")
        log(f"[!] Folder datasets/ sudah ada -> dipindah ke {cadangan.name}")
        shutil.move(str(DATASET_DIR), str(cadangan))

    stat = konversi(root_dataset, kelas_aktif, args.resplit,
                    args.max_train, args.max_valid, args.max_test, args.seed)

    # ---------- tahap D: yaml, grafik, validasi ----------
    tulis_yaml(kelas_aktif)
    plot_distribusi(stat, kelas_aktif)
    validasi(kelas_aktif)

    log("\n=== RINGKASAN ===")
    for split in SPLITS:
        log(f"  {split:<6}: {stat['gambar'][split]:>6} gambar | {stat['bbox'][split]:>6} bbox | "
            f"{stat['kosong'][split]:>4} background")
    log("\n  Instance per kelas:")
    for k in kelas_aktif:
        log(f"    - {k:<18}: {stat['kelas'][k]:>6}")
    if stat["gagal"]:
        log(f"\n  [!] {len(stat['gagal'])} file bermasalah, contoh: {stat['gagal'][:3]}")
    if stat["kelas_asing"]:
        log(f"  [!] Nama kelas asing diabaikan: {dict(stat['kelas_asing'])}")

    log("\n[OK] Dataset siap. Lanjut ke:")
    log("     python scripts_lokal/2_train_yolo.py")
    log(f"     (data.yaml: {DATA_YAML})")


if __name__ == "__main__":
    sys.exit(main())
