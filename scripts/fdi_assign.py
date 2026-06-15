"""
Assign FDI to disease detections via the tooth detector (enumeration YOLO).

For each disease detection (box) -> find the TOOTH box that best 'contains' the
disease box (containment, not IoU — a small lesion sits inside a large tooth).
Take that tooth's FDI. Result: Stage 1 = bbox + FDI + diagnosis (full prediction,
no GT).
"""


def containment(inner, outer):
    """Fraction of 'inner' (disease box) that lies inside 'outer' (tooth box)."""
    x0 = max(inner[0], outer[0]); y0 = max(inner[1], outer[1])
    x1 = min(inner[2], outer[2]); y1 = min(inner[3], outer[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    a_inner = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a_inner


class EnumFDI:
    """Tooth-detector YOLO wrapper. Predicts all teeth + FDI per image."""

    def __init__(self, ckpt, imgsz=1024, conf=0.3):
        from ultralytics import YOLO
        self.model = YOLO(ckpt)
        self.imgsz = imgsz
        self.conf = conf

    def teeth(self, image_path):
        """-> list of (bbox_xyxy, fdi_str, conf)."""
        res = self.model.predict(image_path, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            fdi = str(res.names[int(b.cls)])
            box = [float(v) for v in b.xyxy[0].tolist()]
            out.append((box, fdi, float(b.conf)))
        return out

    @staticmethod
    def assign(disease_box, teeth, min_contain=0.4):
        """FDI of the tooth that best contains the disease box (containment >= min_contain)."""
        best_c, best_fdi = 0.0, None
        for tbox, fdi, _ in teeth:
            c = containment(disease_box, tbox)
            if c > best_c:
                best_c, best_fdi = c, fdi
        return best_fdi if best_c >= min_contain else None
