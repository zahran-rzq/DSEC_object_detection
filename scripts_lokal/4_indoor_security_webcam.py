# =============================================================================
#  Nama : Muhammad Zahran Razzaq
#  NRP  : 5024xxxxxx          <-- WAJIB diisi sebelum dikumpulkan (Nama_NRP.zip)
#  Mata Kuliah : Elektronika Cerdas — ETS Object Detection (Indoor)
#  File : scripts_lokal/4_indoor_security_webcam.py
# =============================================================================
"""
LANGKAH 4 — Aplikasi "Indoor Security Camera" + Object Counting (Slide 3, 29, 37, 38).

Yang dilakukan aplikasi ini:
  * Buka webcam (cv2.VideoCapture) -> deteksi real-time dengan MyYOLOModel.pt.
  * Tampilkan bounding box + label + confidence (results[0].plot()).
  * Panel Object Counter: jumlah objek per kelas di frame saat ini + total
    objek unik yang pernah masuk frame (ByteTrack, bila diaktifkan).
  * AUTO-SAVE  : menyimpan detection_result.jpg saat objek target terdeteksi.
  * AUTO-RECORD: mulai merekam .mp4 saat objek target terdeteksi, berhenti
    otomatis setelah objek hilang beberapa detik (post-roll).

Kontrol keyboard:
    q / ESC : keluar
    s       : simpan snapshot manual
    r       : mulai/hentikan rekaman manual
    c       : reset penghitung unik
    m       : mirror (flip horizontal) on/off

Contoh:
    python scripts_lokal/4_indoor_security_webcam.py
    python scripts_lokal/4_indoor_security_webcam.py --source 1          # webcam eksternal
    python scripts_lokal/4_indoor_security_webcam.py --targets human door
    python scripts_lokal/4_indoor_security_webcam.py --source video.mp4  # uji pakai video
    python scripts_lokal/4_indoor_security_webcam.py --no-display        # headless (Colab/SSH)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Warna BGR per kelas (rapi & konsisten di panel)
WARNA = {
    "bin": (0, 165, 255),
    "chair": (255, 144, 30),
    "door": (60, 200, 60),
    "fire extinguisher": (40, 40, 230),
    "human": (230, 90, 200),
    "shelf": (200, 200, 60),
    "table": (150, 110, 255),
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def buka_sumber(source: str) -> "cv2.VideoCapture":
    import cv2
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(
            f"[X] Tidak bisa membuka sumber video '{source}'.\n"
            "    * Pastikan webcam tidak dipakai aplikasi lain (Zoom/Meet/browser).\n"
            "    * Di Windows coba --source 0 atau --source 1.\n"
            "    * Di Linux cek izin: ls -l /dev/video*"
        )
    return cap


def buat_writer(path: Path, fps: float, ukuran: tuple[int, int]):
    """VideoWriter dengan fallback codec (mp4v -> avc1 -> XVID/.avi)."""
    import cv2
    kandidat = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]
    for codec, ext in kandidat:
        final = path if path.suffix == ext else path.with_suffix(ext)
        writer = cv2.VideoWriter(str(final), cv2.VideoWriter_fourcc(*codec), fps, ukuran)
        if writer.isOpened():
            return writer, final, codec
        writer.release()
    raise RuntimeError("Tidak ada codec VideoWriter yang tersedia di sistem ini.")


def gambar_panel(frame, jumlah: Counter, unik: Counter, fps: float, rekam: bool,
                 target_ada: bool, total_frame: int):
    """Panel semi-transparan berisi Object Counter (Slide 3)."""
    import cv2

    lebar_panel = 300
    # tinggi cukup untuk judul + total + FPS + n baris kelas + status
    tinggi_panel = 124 + 22 * max(len(jumlah), 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + lebar_panel, 10 + tinggi_panel),
                  (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + lebar_panel, 10 + tinggi_panel),
                  (0, 255, 0) if target_ada else (180, 180, 180), 2, cv2.LINE_AA)

    y = 36
    cv2.putText(frame, "INDOOR SECURITY CAMERA", (22, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    y += 24

    total = sum(jumlah.values())
    cv2.putText(frame, f"Objek di frame : {total}", (22, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"FPS: {fps:5.1f}   frame: {total_frame}", (22, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 24

    if jumlah:
        for nama, n in jumlah.most_common():
            warna = WARNA.get(nama, (0, 255, 0))
            cv2.rectangle(frame, (22, y - 9), (34, y + 3), warna, -1)
            cv2.putText(frame, f"{nama}: {n}  (total {unik.get(nama, n)})",
                        (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
    else:
        cv2.putText(frame, "menunggu objek...", (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # indikator rekam
    if rekam:
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 92, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # status target
    cv2.putText(frame, "TARGET TERDETEKSI" if target_ada else "area aman",
                (22, 10 + tinggi_panel - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0) if target_ada else (160, 160, 160), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indoor Security Camera: YOLO real-time + object counter + auto record.")
    parser.add_argument("--model", default=str(ROOT_DIR / "MyYOLOModel.pt"))
    parser.add_argument("--source", default="0", help="0/1 = webcam, atau path file video")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="kelas pemicu alarm, mis. --targets human door "
                             "(default: semua kelas model)")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "hasil_webcam"))
    parser.add_argument("--record-secs", type=float, default=6.0,
                        help="lama rekaman setelah objek terakhir hilang (post-roll)")
    parser.add_argument("--cooldown-secs", type=float, default=2.0,
                        help="jeda minimal antar penyimpanan snapshot otomatis")
    parser.add_argument("--no-record", action="store_true", help="matikan auto-record")
    parser.add_argument("--no-save", action="store_true", help="matikan auto-save gambar")
    parser.add_argument("--no-track", action="store_true",
                        help="matikan ByteTrack (pakai predict biasa)")
    parser.add_argument("--no-display", action="store_true",
                        help="jalan tanpa jendela GUI (untuk Colab/SSH/headless)")
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="berhenti setelah N frame (0 = tanpa batas)")
    args = parser.parse_args()

    import cv2

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(
            f"[X] Model '{model_path}' tidak ditemukan.\n"
            "    Jalankan dulu: python scripts_lokal/2_train_yolo.py\n"
            "    atau: python scripts_lokal/1_setup_dataset_mendeley.py"
        )

    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    nama_kelas = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    targets = set(args.targets) if args.targets else set(nama_kelas.values())
    tak_dikenal = targets - set(nama_kelas.values())
    if tak_dikenal:
        log(f"[!] Kelas target tidak ada di model: {tak_dikenal}")
        log(f"    Kelas model: {list(nama_kelas.values())}")

    log("=" * 62)
    log(" INDOOR SECURITY CAMERA + OBJECT COUNTER")
    log("=" * 62)
    log(f" model   : {model_path}")
    log(f" sumber  : {args.source}")
    log(f" target  : {sorted(targets)}")
    log(f" conf    : {args.conf} | imgsz {args.imgsz} | device {device}")
    log(f" output  : {out_dir}")
    log(" Kontrol : q=keluar  s=snapshot  r=rekam  c=reset  m=mirror")
    log("=" * 62 + "\n")

    cap = buka_sumber(args.source)
    fps_sumber = cap.get(cv2.CAP_PROP_FPS) or 20.0
    if not (1 <= fps_sumber <= 120):
        fps_sumber = 20.0
    lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = None
    jalur_rekaman: Path | None = None
    codec = None
    sedang_rekam = False
    rekaman_selesai: list[Path] = []
    snapshot_selesai: list[Path] = []

    unik = Counter()            # id track -> kelas (Counter kelas unik)
    id_terlihat: dict[int, str] = {}
    target_terakhir = 0.0
    simpan_terakhir = 0.0
    n_frame = 0
    t_mulai = time.time()
    fps_sekarang = 0.0
    frame_terbaik = None
    skor_terbaik = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log("[!] Frame habis / kamera terputus.")
                break
            n_frame += 1
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t0 = time.time()
            if args.no_track:
                hasil = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz,
                                      device=device, verbose=False)[0]
            else:
                hasil = model.track(source=frame, conf=args.conf, imgsz=args.imgsz,
                                    device=device, persist=True, verbose=False,
                                    tracker="bytetrack.yaml")[0]
            dt = max(time.time() - t0, 1e-3)
            fps_instan = min(1.0 / dt, 999.0)
            fps_sekarang = 0.9 * fps_sekarang + 0.1 * fps_instan if fps_sekarang else fps_instan

            # ---------- hitung objek ----------
            jumlah = Counter()
            if hasil.boxes is not None and len(hasil.boxes) > 0:
                kelas_ids = hasil.boxes.cls.tolist()
                ids = hasil.boxes.id.tolist() if hasil.boxes.id is not None else None
                for i, cid in enumerate(kelas_ids):
                    nama = nama_kelas.get(int(cid), str(int(cid)))
                    jumlah[nama] += 1
                    if ids is not None:
                        tid = int(ids[i])
                        if tid not in id_terlihat:
                            id_terlihat[tid] = nama
                            unik[nama] += 1
                            log(f"  + objek baru #{tid:<4} {nama:<18} "
                                f"(t={time.time() - t_mulai:5.1f}s)")

            target_ada = any(n in targets for n in jumlah)
            if target_ada:
                target_terakhir = time.time()

            annotated = hasil.plot()   # bbox + label + confidence
            annotated = gambar_panel(annotated, jumlah, unik, fps_sekarang,
                                     sedang_rekam, target_ada, n_frame)

            # ---------- AUTO-SAVE gambar (Slide 37) ----------
            if target_ada and not args.no_save:
                skor = sum(jumlah.values())
                if skor > skor_terbaik:
                    skor_terbaik = skor
                    frame_terbaik = annotated.copy()
                if time.time() - simpan_terakhir >= args.cooldown_secs:
                    simpan_terakhir = time.time()
                    utama = out_dir / "detection_result.jpg"        # nama wajib
                    cv2.imwrite(str(utama), annotated)
                    arsip = out_dir / f"deteksi_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(arsip), annotated)
                    snapshot_selesai.append(arsip)
                    log(f"  [SNAP] {utama.name} + {arsip.name}  "
                        f"({', '.join(f'{k}={v}' for k, v in jumlah.items())})")

            # ---------- AUTO-RECORD video (Slide 38) ----------
            if not args.no_record:
                if target_ada and not sedang_rekam:
                    stempel = datetime.now().strftime("%Y%m%d_%H%M%S")
                    writer, jalur_rekaman, codec = buat_writer(
                        out_dir / f"rekaman_{stempel}.mp4", fps_sumber,
                        (annotated.shape[1], annotated.shape[0]))
                    sedang_rekam = True
                    log(f"  [REC ] mulai merekam -> {jalur_rekaman.name} (codec {codec})")

                if sedang_rekam and writer is not None:
                    writer.write(annotated)
                    if not target_ada and time.time() - target_terakhir > args.record_secs:
                        writer.release()
                        writer = None
                        sedang_rekam = False
                        if jalur_rekaman:
                            rekaman_selesai.append(jalur_rekaman)
                            ukuran = jalur_rekaman.stat().st_size / 1024 / 1024
                            log(f"  [REC ] selesai -> {jalur_rekaman.name} ({ukuran:.1f} MB)")
                        jalur_rekaman = None

            # ---------- tampilkan ----------
            if not args.no_display:
                cv2.imshow("Indoor Security Camera - ETS Elektronika Cerdas", annotated)
                tombol = cv2.waitKey(1) & 0xFF
                if tombol in (ord("q"), 27):
                    break
                if tombol == ord("s"):
                    path = out_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), annotated)
                    snapshot_selesai.append(path)
                    log(f"  [SNAP] manual -> {path.name}")
                if tombol == ord("r"):
                    if sedang_rekam and writer is not None:
                        writer.release(); writer = None; sedang_rekam = False
                        if jalur_rekaman:
                            rekaman_selesai.append(jalur_rekaman)
                        log("  [REC ] dihentikan manual")
                        jalur_rekaman = None
                    else:
                        writer, jalur_rekaman, _ = buat_writer(
                            out_dir / f"rekaman_manual_{datetime.now():%Y%m%d_%H%M%S}.mp4",
                            fps_sumber, (annotated.shape[1], annotated.shape[0]))
                        sedang_rekam = True
                        log(f"  [REC ] manual -> {jalur_rekaman.name}")
                if tombol == ord("c"):
                    unik.clear(); id_terlihat.clear()
                    log("  [INFO] penghitung unik di-reset")
                if tombol == ord("m"):
                    args.mirror = not args.mirror
                    log(f"  [INFO] mirror = {args.mirror}")

            if args.max_frames and n_frame >= args.max_frames:
                log(f"[i] Batas --max-frames {args.max_frames} tercapai.")
                break

    except KeyboardInterrupt:
        log("\n[!] Dihentikan pengguna (Ctrl+C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            if jalur_rekaman:
                rekaman_selesai.append(jalur_rekaman)
        if not args.no_display:
            cv2.destroyAllWindows()

    # ---------- ringkasan ----------
    durasi = time.time() - t_mulai
    log("\n" + "=" * 62)
    log(" RINGKASAN SESI")
    log("=" * 62)
    log(f" Durasi          : {durasi:.1f} s ({n_frame} frame, "
        f"rata-rata {n_frame / max(durasi, 1e-6):.1f} FPS)")
    log(f" Objek unik      : {dict(unik) if unik else 'tidak ada'}")
    log(f" Snapshot        : {len(snapshot_selesai)} file")
    log(f" Rekaman         : {len(rekaman_selesai)} file")
    for p in rekaman_selesai:
        if p.exists():
            log(f"    - {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    if frame_terbaik is not None:
        path = out_dir / "detection_result_best.jpg"
        cv2.imwrite(str(path), frame_terbaik)
        log(f" Frame terbaik   : {path}")
    log(f"\n[OK] Semua output ada di: {out_dir}")

    # buka folder otomatis (opsional, hanya desktop)
    if sys.platform.startswith("win"):
        try:
            subprocess.Popen(["explorer", str(out_dir)])
        except Exception:
            pass


if __name__ == "__main__":
    main()
