from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, Dataset

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequences[index]


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequence)
        repeated = hidden[-1].unsqueeze(1).repeat(1, sequence.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


@dataclass
class AnomalyResult:
    subsystem: str
    anomaly_score: float
    recommended_action: str
    explanation: ModelExplanation
    affected_entities: List[str]


class AnomalyDetectionEngine(BaseEngine):
    engine_name = "anomaly_detection"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.isolation_forest: Optional[IsolationForest] = None
        self.one_class_svm: Optional[OneClassSVM] = None
        self.sequence_model: Optional[LSTMAutoencoder] = None
        self.feature_columns: List[str] = []
        self.sequence_columns: List[str] = []
        self.sequence_mean: Optional[np.ndarray] = None
        self.sequence_std: Optional[np.ndarray] = None

    def build_feature_frame(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        equipment_telemetry: pd.DataFrame,
    ) -> pd.DataFrame:
        demand = (
            demand_history.assign(timestamp=pd.to_datetime(demand_history["timestamp"]).dt.floor("D"))
            .groupby(["timestamp", "warehouse_id"], as_index=False)["units"]
            .sum()
            .rename(columns={"units": "total_demand"})
        )
        inventory = (
            inventory_history.assign(timestamp=pd.to_datetime(inventory_history["timestamp"]).dt.floor("D"))
            .groupby(["timestamp", "warehouse_id"], as_index=False)
            .agg(
                on_hand=("on_hand", "sum"),
                reserved=("reserved", "sum"),
                damaged=("damaged", "sum"),
                reorder_point=("reorder_point", "sum"),
            )
        )
        telemetry = (
            equipment_telemetry.assign(timestamp=pd.to_datetime(equipment_telemetry["timestamp"]).dt.floor("D"))
            .groupby(["timestamp", "warehouse_id"], as_index=False)
            .agg(
                vibration_rms=("vibration_rms", "mean"),
                motor_temp_c=("motor_temp_c", "mean"),
                acoustic_db=("acoustic_db", "mean"),
                load_factor=("load_factor", "mean"),
            )
        )
        frame = demand.merge(inventory, on=["timestamp", "warehouse_id"], how="inner").merge(
            telemetry,
            on=["timestamp", "warehouse_id"],
            how="inner",
        )
        frame["inventory_gap"] = frame["on_hand"] - frame["reorder_point"]
        frame["damage_ratio"] = frame["damaged"] / np.maximum(frame["on_hand"] + frame["reserved"], 1)
        frame["reservation_pressure"] = frame["reserved"] / np.maximum(frame["on_hand"], 1)
        frame = frame.sort_values(["warehouse_id", "timestamp"]).reset_index(drop=True)
        self.feature_columns = [
            "total_demand",
            "on_hand",
            "reserved",
            "damaged",
            "reorder_point",
            "inventory_gap",
            "damage_ratio",
            "reservation_pressure",
            "vibration_rms",
            "motor_temp_c",
            "acoustic_db",
            "load_factor",
        ]
        return frame

    def build_sequences(self, equipment_telemetry: pd.DataFrame, sequence_length: int = 24) -> np.ndarray:
        telemetry = equipment_telemetry.sort_values(["equipment_id", "timestamp"]).copy()
        self.sequence_columns = [
            "vibration_rms",
            "motor_temp_c",
            "acoustic_db",
            "cycle_count",
            "load_factor",
        ]
        sequences: List[np.ndarray] = []
        for _, group in telemetry.groupby("equipment_id", sort=False):
            values = group[self.sequence_columns].to_numpy(dtype=np.float32)
            if len(values) < sequence_length:
                continue
            for idx in range(sequence_length, len(values) + 1):
                sequences.append(values[idx - sequence_length : idx])
        if not sequences:
            raise ValueError("not enough telemetry to train sequence anomaly detector")
        return np.stack(sequences)

    def train(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        equipment_telemetry: pd.DataFrame,
        epochs: int = 6,
    ) -> TrainResult:
        feature_frame = self.build_feature_frame(demand_history, inventory_history, equipment_telemetry)
        scaled = self.scaler.fit_transform(feature_frame[self.feature_columns])
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.06,
            random_state=42,
        ).fit(scaled)
        self.one_class_svm = OneClassSVM(nu=0.08, kernel="rbf", gamma="scale").fit(scaled)
        sequences = self.build_sequences(equipment_telemetry)
        self.sequence_mean = sequences.reshape(-1, sequences.shape[-1]).mean(axis=0)
        self.sequence_std = sequences.reshape(-1, sequences.shape[-1]).std(axis=0)
        self.sequence_std[self.sequence_std < 1e-6] = 1.0
        sequences = (sequences - self.sequence_mean) / self.sequence_std
        dataset = SequenceDataset(sequences)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.sequence_model = LSTMAutoencoder(input_dim=sequences.shape[-1])
        optimizer = torch.optim.Adam(self.sequence_model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        losses: List[float] = []
        for _ in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                reconstruction = self.sequence_model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.sequence_model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
        model_path = self.save_state(
            self.sequence_model,
            self.artifacts_dir / f"{self.engine_name}_lstm_autoencoder.pt",
            extra={
                "feature_columns": self.feature_columns,
                "sequence_columns": self.sequence_columns,
                "sequence_mean": self.sequence_mean.tolist(),
                "sequence_std": self.sequence_std.tolist(),
            },
        )
        sklearn_artifacts = {
            "scaler": self.scaler,
            "isolation_forest": self.isolation_forest,
            "one_class_svm": self.one_class_svm,
        }
        with (self.artifacts_dir / f"{self.engine_name}_tabular.pkl").open("wb") as handle:
            pickle.dump(sklearn_artifacts, handle)
        metrics = [
            EvaluationMetric(name="autoencoder_loss", value=float(np.mean(losses[-max(len(loader), 1) :]))),
            EvaluationMetric(name="tabular_rows", value=float(len(feature_frame))),
        ]
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=model_path,
            metrics=metrics,
            metadata={"detectors": ["IsolationForest", "LSTMAutoencoder", "OneClassSVM"]},
        )

    def _sequence_score(self, equipment_telemetry: pd.DataFrame) -> Tuple[float, List[str]]:
        if self.sequence_model is None:
            raise RuntimeError("train the sequence anomaly model before scoring")
        sequence_length = 24
        telemetry = equipment_telemetry.sort_values(["equipment_id", "timestamp"]).copy()
        scored_entities: List[str] = []
        reconstruction_errors: List[float] = []
        for equipment_id, group in telemetry.groupby("equipment_id", sort=False):
            values = group[self.sequence_columns].to_numpy(dtype=np.float32)
            if len(values) < sequence_length:
                continue
            if self.sequence_mean is None or self.sequence_std is None:
                raise RuntimeError("sequence normalization stats missing")
            norm = (values[-sequence_length:] - self.sequence_mean) / self.sequence_std
            batch = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                reconstruction = self.sequence_model(batch)
            error = torch.mean((reconstruction - batch) ** 2).item()
            reconstruction_errors.append(error)
            if error > np.mean(reconstruction_errors) + np.std(reconstruction_errors or [0.0]):
                scored_entities.append(str(equipment_id))
        return float(np.mean(reconstruction_errors) if reconstruction_errors else 0.0), scored_entities

    def detect(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        equipment_telemetry: pd.DataFrame,
    ) -> AnomalyResult:
        if self.isolation_forest is None or self.one_class_svm is None:
            raise RuntimeError("train the tabular anomaly detectors before scoring")
        feature_frame = self.build_feature_frame(demand_history, inventory_history, equipment_telemetry)
        recent = feature_frame.tail(48).copy()
        scaled = self.scaler.transform(recent[self.feature_columns])
        iso_scores = -self.isolation_forest.score_samples(scaled)
        svm_scores = -self.one_class_svm.score_samples(scaled)
        seq_score, risky_equipment = self._sequence_score(equipment_telemetry.tail(24 * 8))
        tabular_score = float(np.mean((iso_scores - iso_scores.min()) / (np.ptp(iso_scores) + 1e-6)))
        svm_score = float(np.mean((svm_scores - svm_scores.min()) / (np.ptp(svm_scores) + 1e-6)))
        score = float(np.clip(0.45 * tabular_score + 0.25 * svm_score + 0.30 * min(seq_score, 1.0), 0.0, 1.0))
        latest = recent.iloc[-1]
        feature_importance = {
            "inventory_gap": float(abs(latest["inventory_gap"])),
            "damage_ratio": float(latest["damage_ratio"]),
            "motor_temp_c": float(latest["motor_temp_c"]),
            "load_factor": float(latest["load_factor"]),
        }
        subsystem = "operations"
        if latest["motor_temp_c"] > 70 or risky_equipment:
            subsystem = "equipment"
        elif abs(latest["inventory_gap"]) > latest["reorder_point"] * 0.15 or latest["damage_ratio"] > 0.05:
            subsystem = "inventory"
        actions = {
            "equipment": "Dispatch maintenance diagnostics and reduce routing through the degraded zone.",
            "inventory": "Trigger cycle count audit and quarantine damaged stock for manual verification.",
            "operations": "Escalate to operations control and inspect for process deviations or theft risk.",
        }
        entities = risky_equipment or [str(latest["warehouse_id"])]
        return AnomalyResult(
            subsystem=subsystem,
            anomaly_score=score,
            recommended_action=actions[subsystem],
            explanation=ModelExplanation(
                engine=self.engine_name,
                summary="Ensemble score blends tabular outlier detection with sequence reconstruction error.",
                confidence=float(np.clip(score + 0.2, 0.3, 0.98)),
                feature_contributions=feature_importance,
                evidence=[
                    f"isolation_mean={tabular_score:.3f}",
                    f"svm_mean={svm_score:.3f}",
                    f"sequence_score={seq_score:.3f}",
                ],
            ),
            affected_entities=entities,
        )
