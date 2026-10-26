HASIL ETS ELEKTRONIKA CERDAS — OBJECT DETECTION (INDOOR)
Nama : Muhammad Zahran Razzaq
NRP  : 5024xxxxxx

Model pre-trained : yolo12n.pt
Kelas (7)            : ['bin', 'chair', 'door', 'fire extinguisher', 'human', 'shelf', 'table']
Training          : 30 epoch (patience 15), imgsz 640, batch 16
Durasi training   : 16.7 menit

VALIDATION -> P=0.7830  R=0.7882  mAP50=0.8306  mAP50-95=0.5807
TEST       -> P=0.8533  R=0.7474  mAP50=0.8374  mAP50-95=0.5941

Isi folder:
  MyYOLOModel.pt           : model hasil transfer learning
  data.yaml                : konfigurasi dataset (path absolut)
  class_distribution.png   : distribusi kelas (Slide 31)
  map_per_class.png        : mAP per kelas (Slide 32)
  ETS_results/             : results.png, training_curves.png, confusion_matrix.png,
                             val_batch0_labels.jpg, val_batch0_pred.jpg,
                             perbandingan_labels_vs_pred.jpg, compare_coco_vs_custom.png,
                             metrics_summary.txt, per-class CSV
  hasil_inferensi/         : detection_result.jpg, security_camera_recording.mp4, snapshot
  metrics.json             : metrik dalam format mesin
