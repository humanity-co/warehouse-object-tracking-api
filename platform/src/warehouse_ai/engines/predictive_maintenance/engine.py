from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult


class MaintenanceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, rul: np.ndarray, fail_prob: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.rul = torch.tensor(rul, dtype=torch.float32)
        self.fail_prob = torch.tensor(fail_prob, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.rul[index], self.fail_prob[index]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = self.downsample(x)
        min_len = min(out.shape[-1], residual.shape[-1])
        return out[..., :min_len] + residual[..., :min_len]


class MaintenanceTCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalBlock(hidden_dim, hidden_dim, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rul_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.tcn(sequence.transpose(1, 2))
        pooled = self.pool(features).squeeze(-1)
        rul = F.softplus(self.rul_head(pooled)).squeeze(-1)
        risk = torch.sigmoid(self.risk_head(pooled)).squeeze(-1)
        return rul, risk


@dataclass
class MaintenanceRecommendation:
    equipment_id: str
    warehouse_id: str
    remaining_useful_life_hours: float
    failure_probability_72h: float
    priority_score: float
    degraded_zone: str
    explanation: ModelExplanation


class PredictiveMaintenanceEngine(BaseEngine):
    engine_name = "predictive_maintenance"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[MaintenanceTCN] = None
        self.feature_columns = [
            "vibration_rms",
            "motor_temp_c",
            "acoustic_db",
            "cycle_count",
            "load_factor",
        ]
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def _degradation_score(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            0.35 * frame["vibration_rms"].to_numpy(dtype=float)
            + 0.30 * frame["motor_temp_c"].to_numpy(dtype=float) / 100.0
            + 0.20 * frame["acoustic_db"].to_numpy(dtype=float) / 100.0
            + 0.15 * frame["load_factor"].to_numpy(dtype=float)
        )

    def build_training_data(
        self,
        equipment_telemetry: pd.DataFrame,
        sequence_length: int = 24,
        failure_horizon: int = 72,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        telemetry = equipment_telemetry.sort_values(["equipment_id", "timestamp"]).copy()
        sequences: List[np.ndarray] = []
        rul_targets: List[float] = []
        fail_targets: List[float] = []
        for _, group in telemetry.groupby("equipment_id", sort=False):
            features = group[self.feature_columns].to_numpy(dtype=np.float32)
            degradation = self._degradation_score(group)
            failure_points = np.where(degradation > np.quantile(degradation, 0.92))[0]
            failure_index = int(failure_points[0]) if len(failure_points) else len(group) - 1
            if len(features) < sequence_length + 1:
                continue
            for idx in range(sequence_length, len(features)):
                sequences.append(features[idx - sequence_length : idx])
                remaining = max(1.0, float(failure_index - idx))
                rul_targets.append(remaining)
                fail_targets.append(1.0 if remaining <= failure_horizon else 0.0)
        if not sequences:
            raise ValueError("not enough telemetry to build maintenance dataset")
        return np.stack(sequences), np.array(rul_targets, dtype=np.float32), np.array(fail_targets, dtype=np.float32)

    def train(self, equipment_telemetry: pd.DataFrame, epochs: int = 8) -> TrainResult:
        sequences, rul_targets, fail_targets = self.build_training_data(equipment_telemetry)
        self.mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
        self.std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        sequences = (sequences - self.mean) / self.std
        dataset = MaintenanceDataset(sequences, np.log1p(rul_targets), fail_targets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model = MaintenanceTCN(input_dim=sequences.shape[-1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        mse = nn.MSELoss()
        bce = nn.BCELoss()
        epoch_losses: List[float] = []
        for _ in range(epochs):
            for batch, rul_batch, fail_batch in loader:
                optimizer.zero_grad()
                pred_rul, pred_fail = self.model(batch)
                loss = mse(pred_rul, rul_batch) + 0.7 * bce(pred_fail, fail_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
        artifact_path = self.save_state(
            self.model,
            self.artifacts_dir / f"{self.engine_name}_tcn.pt",
            extra={
                "feature_columns": self.feature_columns,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            },
        )
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=[
                EvaluationMetric(name="training_loss", value=float(np.mean(epoch_losses[-max(len(loader), 1) :]))),
                EvaluationMetric(name="samples", value=float(len(dataset))),
            ],
            metadata={"models": ["TCN", "survival_head"]},
        )

    def recommend(self, equipment_telemetry: pd.DataFrame) -> List[MaintenanceRecommendation]:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("train the predictive maintenance model before inference")
        telemetry = equipment_telemetry.sort_values(["equipment_id", "timestamp"]).copy()
        recommendations: List[MaintenanceRecommendation] = []
        for equipment_id, group in telemetry.groupby("equipment_id", sort=False):
            if len(group) < 24:
                continue
            sequence = group[self.feature_columns].to_numpy(dtype=np.float32)[-24:]
            norm = (sequence - self.mean) / self.std
            batch = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred_rul, pred_fail = self.model(batch)
            rul_hours = float(np.expm1(pred_rul.item()))
            fail_prob = float(pred_fail.item())
            latest = group.iloc[-1]
            priority = float(np.clip((1.0 / max(rul_hours, 1.0)) * 80 + fail_prob * 30, 0.0, 100.0))
            recommendations.append(
                MaintenanceRecommendation(
                    equipment_id=str(equipment_id),
                    warehouse_id=str(latest["warehouse_id"]),
                    remaining_useful_life_hours=rul_hours,
                    failure_probability_72h=fail_prob,
                    priority_score=priority,
                    degraded_zone=f"{latest['warehouse_id']}-maintenance-buffer",
                    explanation=ModelExplanation(
                        engine=self.engine_name,
                        summary="TCN sequence patterns indicate degradation trajectory and short-horizon failure risk.",
                        confidence=float(np.clip(0.55 + fail_prob * 0.4, 0.4, 0.98)),
                        feature_contributions={
                            "vibration_rms": float(latest["vibration_rms"]),
                            "motor_temp_c": float(latest["motor_temp_c"]),
                            "acoustic_db": float(latest["acoustic_db"]),
                            "load_factor": float(latest["load_factor"]),
                        },
                        evidence=[
                            f"predicted_rul_hours={rul_hours:.2f}",
                            f"failure_probability_72h={fail_prob:.3f}",
                        ],
                    ),
                )
            )
        return sorted(recommendations, key=lambda item: item.priority_score, reverse=True)
