from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult

try:  # pragma: no cover - optional production adapter
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:  # pragma: no cover - optional production adapter
    from transformers import DetrImageProcessor, DetrForObjectDetection
except ImportError:  # pragma: no cover - optional dependency
    DetrImageProcessor = None
    DetrForObjectDetection = None


COLOR_MAP = {
    "beverage": np.array([52, 152, 219], dtype=np.uint8),
    "snack": np.array([241, 196, 15], dtype=np.uint8),
    "health": np.array([46, 204, 113], dtype=np.uint8),
    "electronics": np.array([231, 76, 60], dtype=np.uint8),
    "household": np.array([155, 89, 182], dtype=np.uint8),
}


class DamageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class DamageClassifierCNN(nn.Module):
    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


@dataclass
class VisionFrame:
    frame_id: str
    image: np.ndarray
    warehouse_id: str
    zone_id: str
    expected_sku_id: str
    expected_count: int
    barcode_text: str
    severity_label: int


@dataclass
class VisionDecision:
    frame_id: str
    detected_count: int
    count_match: bool
    misplaced: bool
    damage_severity: str
    quarantine: bool
    explanation: ModelExplanation
    metadata: Dict[str, Any]  # New field for UI-specific data


class SimulatedBarcodeOCR:
    """Simulation layer: production replacement is OCR/barcode hardware or a cloud OCR service."""

    def decode(self, frame: VisionFrame) -> str:
        return frame.barcode_text


class ColorBlobDetector:
    """Local detector fallback when YOLOv8/DETR adapters are unavailable."""

    def detect(self, frame: VisionFrame) -> Dict[str, Any]:
        mask = np.any(frame.image < 250, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return {"count": 0, "bbox": None, "confidence": 0.0}
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        count = max(1, int(np.round((mask.sum() / 1000.0))))
        return {"count": count, "bbox": bbox, "confidence": 0.61}


class YoloV8Adapter:
    def __init__(self, weights: str = "yolov8n.pt") -> None:
        self.weights = weights
        self.model = None

    def available(self) -> bool:
        self._ensure_loaded()
        return self.model is not None

    def _ensure_loaded(self) -> None:
        if self.model is not None or YOLO is None:
            return
        try:
            self.model = YOLO(self.weights)
        except Exception:  # pragma: no cover - optional dependency bootstrap
            self.model = None

    def detect(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if self.model is None:
            raise RuntimeError("YOLOv8 dependency not installed")
        results = self.model.predict(images, verbose=False)
        outputs: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, "xyxy") else np.empty((0, 4))
            outputs.append(
                {
                    "count": int(len(boxes)),
                    "bbox": boxes.tolist()[0] if len(boxes) else None,
                    "confidence": float(result.boxes.conf.mean().item()) if len(boxes) else 0.0,
                }
            )
        return outputs


class DetrAdapter:
    def __init__(self, model_name: str = "facebook/detr-resnet-50") -> None:
        self.model_name = model_name
        self.processor = None
        self.model = None

    def available(self) -> bool:
        self._ensure_loaded()
        return self.processor is not None and self.model is not None

    def _ensure_loaded(self) -> None:
        if (self.processor is not None and self.model is not None) or DetrImageProcessor is None or DetrForObjectDetection is None:
            return
        try:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
        except Exception:  # pragma: no cover - optional dependency bootstrap
            self.processor = None
            self.model = None

    def detect(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if self.processor is None or self.model is None:
            raise RuntimeError("DETR dependency not installed")
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[images[0].shape[0], images[0].shape[1]]] * len(images)),
            threshold=0.6,
        )
        parsed = []
        for result in results:
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            parsed.append(
                {
                    "count": int(len(boxes)),
                    "bbox": boxes.tolist()[0] if len(boxes) else None,
                    "confidence": float(scores.mean()) if len(scores) else 0.0,
                }
            )
        return parsed


class ComputerVisionEngine(BaseEngine):
    engine_name = "computer_vision"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.damage_model: Optional[DamageClassifierCNN] = None
        self.ocr = SimulatedBarcodeOCR()
        self.yolo = YoloV8Adapter()
        self.detr = DetrAdapter()
        self.detector = ColorBlobDetector()

    def _draw_box(self, image: np.ndarray, top_left: Tuple[int, int], size: Tuple[int, int], color: np.ndarray) -> None:
        x0, y0 = top_left
        w, h = size
        image[y0 : y0 + h, x0 : x0 + w] = color
        image[y0 : y0 + 2, x0 : x0 + w] = 20
        image[y0 + h - 2 : y0 + h, x0 : x0 + w] = 20
        image[y0 : y0 + h, x0 : x0 + 2] = 20
        image[y0 : y0 + h, x0 + w - 2 : x0 + w] = 20

    def generate_synthetic_frames(self, sku_catalog: pd.DataFrame, samples: int = 240) -> List[VisionFrame]:
        frames: List[VisionFrame] = []
        categories = sku_catalog.set_index("sku_id")["category"].to_dict()
        for idx in range(samples):
            sku_row = sku_catalog.iloc[idx % len(sku_catalog)]
            image = np.full((96, 96, 3), 255, dtype=np.uint8)
            count = int(np.random.randint(1, 5))
            severity = int(np.random.choice([0, 1, 2], p=[0.65, 0.23, 0.12]))
            for box_idx in range(count):
                top_left = (10 + box_idx * 18, 24 + (box_idx % 2) * 10)
                self._draw_box(
                    image,
                    top_left=top_left,
                    size=(16, 20),
                    color=COLOR_MAP[categories[sku_row["sku_id"]]],
                )
            if severity > 0:
                scratches = 3 + severity * 4
                for _ in range(scratches):
                    x = int(np.random.randint(6, 88))
                    y = int(np.random.randint(6, 88))
                    image[max(0, y - 1) : min(96, y + 1), max(0, x - 8) : min(96, x + 8)] = np.array([180, 40, 40], dtype=np.uint8)
            frames.append(
                VisionFrame(
                    frame_id=f"frame-{idx:05d}",
                    image=image,
                    warehouse_id=f"WH-{(idx % 3) + 1:02d}",
                    zone_id=f"WH-{(idx % 3) + 1:02d}-Z{(idx % 4) + 1}",
                    expected_sku_id=str(sku_row["sku_id"]),
                    expected_count=count,
                    barcode_text=f"{sku_row['sku_id']}-BATCH-{idx:04d}",
                    severity_label=severity,
                )
            )
        return frames

    def train(self, sku_catalog: pd.DataFrame, epochs: int = 6) -> TrainResult:
        frames = self.generate_synthetic_frames(sku_catalog)
        images = np.stack([frame.image for frame in frames])
        labels = np.array([frame.severity_label for frame in frames], dtype=np.int64)
        dataset = DamageDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.damage_model = DamageClassifierCNN(classes=3)
        optimizer = torch.optim.Adam(self.damage_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        losses: List[float] = []
        for _ in range(epochs):
            for batch, label in loader:
                optimizer.zero_grad()
                logits = self.damage_model(batch)
                loss = criterion(logits, label)
                loss.backward()
                nn.utils.clip_grad_norm_(self.damage_model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
        artifact_path = self.save_state(
            self.damage_model,
            self.artifacts_dir / f"{self.engine_name}_damage_cnn.pt",
            extra={"classes": ["none", "minor", "major"]},
        )
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=[EvaluationMetric(name="damage_training_loss", value=float(np.mean(losses[-max(len(loader), 1) :])))],
            metadata={
                "detector_primary": "yolov8" if self.yolo.available() else "simulation_fallback",
                "detector_secondary": "detr" if self.detr.available() else "simulation_fallback",
            },
        )

    def _run_detector(self, frames: List[VisionFrame]) -> List[Dict[str, Any]]:
        images = [frame.image for frame in frames]
        if self.yolo.available():
            return self.yolo.detect(images)
        if self.detr.available():
            return self.detr.detect(images)
        return [self.detector.detect(frame) for frame in frames]

    def infer_stream(self, frames: List[VisionFrame], batch_size: int = 8) -> List[VisionDecision]:
        if self.damage_model is None:
            raise RuntimeError("train the damage classifier before stream inference")
        decisions: List[VisionDecision] = []
        severity_names = {0: "none", 1: "minor", 2: "major"}
        self.damage_model.eval()
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            detector_outputs = self._run_detector(batch_frames)
            image_tensor = (
                torch.tensor(np.stack([frame.image for frame in batch_frames]), dtype=torch.float32)
                .permute(0, 3, 1, 2)
                / 255.0
            )
            with torch.no_grad():
                damage_logits = self.damage_model(image_tensor)
                damage_scores = torch.softmax(damage_logits, dim=-1)
            for idx, frame in enumerate(batch_frames):
                detected = detector_outputs[idx]
                damage_class = int(torch.argmax(damage_scores[idx]).item())
                barcode = self.ocr.decode(frame)
                count_match = abs(int(detected["count"]) - frame.expected_count) <= 1
                misplaced = frame.expected_sku_id not in barcode
                quarantine = damage_class >= 2
                decisions.append(
                    VisionDecision(
                        frame_id=frame.frame_id,
                        detected_count=int(detected["count"]),
                        count_match=count_match,
                        misplaced=misplaced,
                        damage_severity=severity_names[damage_class],
                        quarantine=quarantine,
                        explanation=ModelExplanation(
                            engine=self.engine_name,
                            summary="Perception result fuses object detection, barcode decoding, and damage severity classification.",
                            confidence=float(np.clip(float(detected["confidence"]) * 0.8 + damage_scores[idx, damage_class].item() * 0.2, 0.3, 0.98)),
                            feature_contributions={
                                "detected_count": float(detected["count"]),
                                "expected_count": float(frame.expected_count),
                                "damage_probability": float(damage_scores[idx, damage_class].item()),
                            },
                            evidence=[
                                f"barcode={barcode}",
                                f"detector_confidence={detected['confidence']:.3f}",
                            ],
                        ),
                        metadata={
                            "sku_id": frame.expected_sku_id,
                            "barcode": barcode,
                            "bbox": detected["bbox"],
                            "node_id": "JETSON-ORIN-MG07",
                            "camera_id": "4K-CAM-WH01-A",
                            "status": "QUARANTINE_REQUIRED" if quarantine else "CLEAR",
                        }
                    )
                )
        return decisions
