MODEL: YOLOV8m.PT

FIRST TRIAL:

konvert dentex to yolo:
{
    Images: 705 (disease 678 + background 27) | train 599 | val 106
    train boxes: {'Impacted': 507, 'Caries': 1847, 'Periapical Lesion': 135, 'Deep Caries': 487}
    val   boxes: {'Impacted': 97, 'Caries': 342, 'Periapical Lesion': 23, 'Deep Caries': 91}
    OK: 3529 boxes written. Dataset YAML: /content/yolo/dentex.yaml
    --- contoh label ---
    1 0.654290 0.461661 0.051738 0.245924
    1 0.274418 0.636433 0.070175 0.227964
}

parameter:
{
    epochs=80, imgsz=1024, batch=16,
}

time train yolov8m:

start: 1.19 PM

epoch 1
mAP50 - 0.283
mAP50-95 - 0.183





