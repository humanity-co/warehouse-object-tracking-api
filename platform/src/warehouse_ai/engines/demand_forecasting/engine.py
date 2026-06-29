from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult
from warehouse_ai.mlops.drift import DistributionShiftDetector


HORIZON_STEPS = [1, 7, 30, 90]
QUANTILES = [0.1, 0.5, 0.9]


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return std


def quantile_loss(predictions: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for idx, quantile in enumerate(quantiles):
        errors = target - predictions[:, :, idx]
        losses.append(torch.maximum((quantile - 1) * errors, quantile * errors).mean())
    return torch.stack(losses).mean()


class ForecastDataset(Dataset):
    def __init__(
        self,
        past: np.ndarray,
        future: np.ndarray,
        static: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.past = torch.tensor(past, dtype=torch.float32)
        self.future = torch.tensor(future, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.past.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.past[index], self.future[index], self.static[index], self.target[index]


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = F.elu(self.fc1(x))
        hidden = self.fc2(hidden)
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.weight_net = nn.Linear(feature_dim, feature_dim)
        self.projections = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_dim) for _ in range(feature_dim)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_net(x), dim=-1)
        transformed = []
        for idx in range(self.feature_dim):
            transformed.append(self.projections[idx](x[..., idx : idx + 1]))
        stacked = torch.stack(transformed, dim=-2)
        selected = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return selected, weights


class TemporalFusionTransformerLite(nn.Module):
    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        hidden_dim: int,
        horizons: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.past_selector = VariableSelectionNetwork(past_dim, hidden_dim)
        self.future_selector = VariableSelectionNetwork(future_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.fusion = GatedResidualNetwork(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, quantiles)
        self.horizons = horizons

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_ctx = self.static_encoder(static)
        selected_past, past_weights = self.past_selector(past)
        encoded = self.temporal_encoder(selected_past + static_ctx.unsqueeze(1))
        selected_future, _ = self.future_selector(future)
        attended, attention_weights = self.cross_attention(
            query=selected_future + static_ctx.unsqueeze(1),
            key=encoded,
            value=encoded,
        )
        fused = self.fusion(attended + selected_future + static_ctx.unsqueeze(1))
        return self.head(fused), past_weights.mean(dim=1)


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        hidden_dim: int,
        horizons: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        self.static_encoder = nn.Linear(static_dim, hidden_dim)
        self.past_proj = nn.Linear(past_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.future_proj = nn.Linear(future_dim, hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, quantiles),
        )
        self.horizons = horizons

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        static_ctx = torch.relu(self.static_encoder(static))
        past_hidden = torch.relu(self.past_proj(past)) + static_ctx.unsqueeze(1)
        outputs, _ = self.lstm(past_hidden)
        attention_scores = torch.softmax(self.attention(outputs).squeeze(-1), dim=-1)
        context = torch.sum(outputs * attention_scores.unsqueeze(-1), dim=1)
        future_hidden = torch.relu(self.future_proj(future))
        repeated_context = context.unsqueeze(1).expand(-1, self.horizons, -1)
        repeated_static = static_ctx.unsqueeze(1).expand(-1, self.horizons, -1)
        fused = torch.cat([repeated_context, future_hidden, repeated_static], dim=-1)
        return self.head(fused), attention_scores


@dataclass
class ForecastOutput:
    sku_id: str
    warehouse_id: str
    horizons: Dict[str, Dict[str, float]]
    explanation: ModelExplanation
    model_name: str
    drift_detected: bool


class DemandForecastEngine(BaseEngine):
    engine_name = "demand_forecasting"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[nn.Module] = None
        self.preprocessing: Dict[str, Any] = {}
        self.model_name = "temporal_fusion_transformer"
        self.detector = DistributionShiftDetector()
        self.try_load()

    def try_load(self) -> bool:
        artifact = self.find_latest_artifact(self.artifacts_dir)
        if not artifact:
            # print(f"No artifact found for {self.engine_name}", file=sys.stderr)
            return False
        try:
            print(f"Loading {self.engine_name} model from {artifact.name}...", file=sys.stderr)
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            extra = payload.get("extra", {})
            self.model_name = extra.get("model_name", self.model_name)
            self.preprocessing = extra.get("preprocessing", {})
            self.preprocessing["past_cols"] = extra.get("past_cols", [])
            self.preprocessing["future_cols"] = extra.get("future_cols", [])
            self.preprocessing["static_cols"] = extra.get("static_cols", [])
            
            self.model = self._create_model(
                past_dim=len(self.preprocessing["past_cols"]),
                future_dim=len(self.preprocessing["future_cols"]),
                static_dim=len(self.preprocessing["static_cols"]),
                sparse_mode=(self.model_name == "attention_bilstm"),
            )
            self.model.load_state_dict(payload["state_dict"])
            print(f"Successfully loaded {self.engine_name} model.", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Error loading {self.engine_name} model: {e}", file=sys.stderr)
            return False

    def build_feature_frame(
        self,
        demand_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
        frame = demand_history.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values(["warehouse_id", "sku_id", "timestamp"]).reset_index(drop=True)
        frame["day_idx"] = (
            frame["timestamp"] - frame["timestamp"].min()
        ).dt.days.astype(int)
        frame["dow_sin"] = np.sin(2 * np.pi * frame["timestamp"].dt.dayofweek / 7.0)
        frame["dow_cos"] = np.cos(2 * np.pi * frame["timestamp"].dt.dayofweek / 7.0)
        frame["month_sin"] = np.sin(2 * np.pi * frame["timestamp"].dt.month / 12.0)
        frame["month_cos"] = np.cos(2 * np.pi * frame["timestamp"].dt.month / 12.0)
        group_cols = ["warehouse_id", "sku_id"]
        frame["lag_1"] = frame.groupby(group_cols)["units"].shift(1)
        frame["lag_7"] = frame.groupby(group_cols)["units"].shift(7)
        frame["lag_14"] = frame.groupby(group_cols)["units"].shift(14)
        frame["rolling_7_mean"] = (
            frame.groupby(group_cols)["units"].rolling(7, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        )
        frame["rolling_7_std"] = (
            frame.groupby(group_cols)["units"].rolling(7, min_periods=2).std().reset_index(level=group_cols, drop=True).fillna(0.0)
        )
        frame["rolling_28_mean"] = (
            frame.groupby(group_cols)["units"].rolling(28, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        )
        frame["rolling_28_std"] = (
            frame.groupby(group_cols)["units"].rolling(28, min_periods=2).std().reset_index(level=group_cols, drop=True).fillna(0.0)
        )
        frame = frame.fillna(0.0)
        static = sku_catalog.copy()
        categorical = ["category", "storage_zone", "supplier_id"]
        static = pd.get_dummies(static, columns=categorical, dtype=float)
        frame = frame.merge(static, on="sku_id", how="left")
        static_cols = [
            column
            for column in frame.columns
            if column.startswith("category_")
            or column.startswith("storage_zone_")
            or column.startswith("supplier_id_")
            or column in {
                "lead_time_days",
                "unit_cost",
                "shelf_life_days",
                "length_cm",
                "width_cm",
                "height_cm",
                "weight_kg",
                "service_level_target",
                "base_demand",
            }
        ]
        past_cols = [
            "units",
            "promotion_flag",
            "external_signal",
            "price_index",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_7_mean",
            "rolling_7_std",
            "rolling_28_mean",
            "rolling_28_std",
        ]
        future_cols = [
            "promotion_flag",
            "external_signal",
            "price_index",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]
        return frame, past_cols, future_cols, static_cols

    def _build_samples(
        self,
        feature_frame: pd.DataFrame,
        past_cols: List[str],
        future_cols: List[str],
        static_cols: List[str],
        sequence_length: int = 30,
    ) -> Dict[str, Any]:
        past_sequences: List[np.ndarray] = []
        future_sequences: List[np.ndarray] = []
        static_vectors: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []
        max_horizon = max(HORIZON_STEPS)
        for (warehouse_id, sku_id), group in feature_frame.groupby(["warehouse_id", "sku_id"], sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < sequence_length + max_horizon:
                continue
            for anchor in range(sequence_length - 1, len(group) - max_horizon):
                window = group.iloc[anchor - sequence_length + 1 : anchor + 1]
                future_known = np.stack(
                    [group.iloc[anchor + horizon][future_cols].to_numpy(dtype=np.float32) for horizon in HORIZON_STEPS]
                )
                target = np.array(
                    [
                        np.log1p(float(group.iloc[anchor + horizon]["units"]))
                        for horizon in HORIZON_STEPS
                    ],
                    dtype=np.float32,
                )
                past_sequences.append(window[past_cols].to_numpy(dtype=np.float32))
                future_sequences.append(future_known)
                static_vectors.append(group.iloc[anchor][static_cols].to_numpy(dtype=np.float32))
                targets.append(target)
                metadata.append(
                    {
                        "warehouse_id": warehouse_id,
                        "sku_id": sku_id,
                        "timestamp": group.iloc[anchor]["timestamp"],
                    }
                )
        if not targets:
            raise ValueError("not enough history to build training samples")

        past = np.stack(past_sequences)
        future = np.stack(future_sequences)
        static = np.stack(static_vectors)
        target = np.stack(targets)
        return {
            "past": past,
            "future": future,
            "static": static,
            "target": target,
            "metadata": metadata,
        }

    def _normalize(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        past = samples["past"]
        future = samples["future"]
        static = samples["static"]
        past_mean = past.reshape(-1, past.shape[-1]).mean(axis=0)
        future_mean = future.reshape(-1, future.shape[-1]).mean(axis=0)
        static_mean = static.mean(axis=0)
        samples["past"] = (past - past_mean) / _safe_std(past.reshape(-1, past.shape[-1]))
        samples["future"] = (future - future_mean) / _safe_std(future.reshape(-1, future.shape[-1]))
        samples["static"] = (static - static_mean) / _safe_std(static)
        self.preprocessing = {
            "past_mean": past_mean,
            "past_std": _safe_std(past.reshape(-1, past.shape[-1])),
            "future_mean": future_mean,
            "future_std": _safe_std(future.reshape(-1, future.shape[-1])),
            "static_mean": static_mean,
            "static_std": _safe_std(static),
            "sequence_length": past.shape[1],
        }
        return samples

    def _create_model(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        sparse_mode: bool,
    ) -> nn.Module:
        if sparse_mode:
            self.model_name = "attention_bilstm"
            return AttentionBiLSTM(past_dim, future_dim, static_dim, hidden_dim=48, horizons=len(HORIZON_STEPS), quantiles=len(QUANTILES))
        self.model_name = "temporal_fusion_transformer"
        return TemporalFusionTransformerLite(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            hidden_dim=64,
            horizons=len(HORIZON_STEPS),
            quantiles=len(QUANTILES),
        )

    def train(
        self,
        demand_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        epochs: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> TrainResult:
        feature_frame, past_cols, future_cols, static_cols = self.build_feature_frame(demand_history, sku_catalog)
        samples = self._build_samples(feature_frame, past_cols, future_cols, static_cols)
        sparse_mode = len(samples["target"]) < 256
        samples = self._normalize(samples)
        indices = np.arange(len(samples["target"]))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
        train_data = ForecastDataset(
            samples["past"][train_idx],
            samples["future"][train_idx],
            samples["static"][train_idx],
            samples["target"][train_idx],
        )
        val_data = ForecastDataset(
            samples["past"][val_idx],
            samples["future"][val_idx],
            samples["static"][val_idx],
            samples["target"][val_idx],
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.model = self._create_model(
            past_dim=samples["past"].shape[-1],
            future_dim=samples["future"].shape[-1],
            static_dim=samples["static"].shape[-1],
            sparse_mode=sparse_mode,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_loss = float("inf")
        best_state: Dict[str, Any] = {}
        for _ in range(epochs):
            self.model.train()
            for past_batch, future_batch, static_batch, target_batch in train_loader:
                optimizer.zero_grad()
                prediction, _ = self.model(past_batch, future_batch, static_batch)
                loss = quantile_loss(prediction, target_batch, QUANTILES)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            val_loss = self._evaluate_loader(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {"state_dict": self.model.state_dict()}
        self.model.load_state_dict(best_state["state_dict"])
        metrics = self.evaluate(val_loader)
        artifact_path = self.save_state(
            self.model,
            self.artifacts_dir / f"{self.engine_name}_{self.model_name}.pt",
            extra={
                "model_name": self.model_name,
                "past_cols": past_cols,
                "future_cols": future_cols,
                "static_cols": static_cols,
                "preprocessing": self.preprocessing,
            },
        )
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=metrics,
            metadata={"model_name": self.model_name, "sparse_mode": sparse_mode},
        )

    def _evaluate_loader(self, loader: DataLoader) -> float:
        if self.model is None:
            raise RuntimeError("model not initialized")
        self.model.eval()
        losses = []
        with torch.no_grad():
            for past_batch, future_batch, static_batch, target_batch in loader:
                prediction, _ = self.model(past_batch, future_batch, static_batch)
                losses.append(quantile_loss(prediction, target_batch, QUANTILES).item())
        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, loader: DataLoader) -> List[EvaluationMetric]:
        if self.model is None:
            raise RuntimeError("model not initialized")
        preds = []
        p10s = []
        p90s = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for past_batch, future_batch, static_batch, target_batch in loader:
                prediction, _ = self.model(past_batch, future_batch, static_batch)
                preds.append(torch.expm1(prediction[:, :, 1]).cpu().numpy())
                p10s.append(torch.expm1(prediction[:, :, 0]).cpu().numpy())
                p90s.append(torch.expm1(prediction[:, :, 2]).cpu().numpy())
                targets.append(torch.expm1(target_batch).cpu().numpy())
        pred = np.concatenate(preds)
        p10 = np.concatenate(p10s)
        p90 = np.concatenate(p90s)
        actual = np.concatenate(targets)
        mae = float(np.mean(np.abs(pred - actual)))
        wape = float(np.sum(np.abs(pred - actual)) / max(np.sum(actual), 1.0))
        p90_spread = float(np.mean(np.maximum(0.0, p90 - p10)))
        return [
            EvaluationMetric(name="mae", value=mae),
            EvaluationMetric(name="wape", value=wape),
            EvaluationMetric(name="pinball", value=self._evaluate_loader(loader)),
            EvaluationMetric(name="forecast_spread_proxy", value=p90_spread),
        ]

    def _prepare_single_sample(
        self,
        feature_frame: pd.DataFrame,
        past_cols: List[str],
        future_cols: List[str],
        static_cols: List[str],
        sku_id: str,
        warehouse_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group = feature_frame[
            (feature_frame["sku_id"] == sku_id) & (feature_frame["warehouse_id"] == warehouse_id)
        ].sort_values("timestamp")
        if len(group) < self.preprocessing["sequence_length"] + max(HORIZON_STEPS):
            raise ValueError("insufficient history for direct inference")
        anchor = len(group) - max(HORIZON_STEPS) - 1
        window = group.iloc[
            anchor - self.preprocessing["sequence_length"] + 1 : anchor + 1
        ][past_cols].to_numpy(dtype=np.float32)
        future = np.stack(
            [group.iloc[anchor + horizon][future_cols].to_numpy(dtype=np.float32) for horizon in HORIZON_STEPS]
        )
        static_vector = group.iloc[anchor][static_cols].to_numpy(dtype=np.float32)
        norm_past = (window - self.preprocessing["past_mean"]) / self.preprocessing["past_std"]
        norm_future = (future - self.preprocessing["future_mean"]) / self.preprocessing["future_std"]
        norm_static = (static_vector - self.preprocessing["static_mean"]) / self.preprocessing["static_std"]
        return (
            torch.tensor(norm_past, dtype=torch.float32).unsqueeze(0),
            torch.tensor(norm_future, dtype=torch.float32).unsqueeze(0),
            torch.tensor(norm_static, dtype=torch.float32).unsqueeze(0),
        )

    def cold_start_forecast(
        self,
        sku_id: str,
        warehouse_id: str,
        demand_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
    ) -> ForecastOutput:
        catalog = sku_catalog.copy()
        target_row = catalog[catalog["sku_id"] == sku_id]
        if target_row.empty:
            raise KeyError("unknown sku for cold-start forecast")
        target = target_row.iloc[0]
        peers = catalog[
            (catalog["category"] == target["category"])
            & (catalog["storage_zone"] == target["storage_zone"])
        ].copy()
        peers["distance"] = (
            (peers["unit_cost"] - target["unit_cost"]).abs()
            + (peers["lead_time_days"] - target["lead_time_days"]).abs()
            + (peers["base_demand"] - target["base_demand"]).abs()
        )
        peer_ids = peers.sort_values("distance")["sku_id"].head(5).tolist()
        peer_demand = demand_history[
            (demand_history["sku_id"].isin(peer_ids))
            & (demand_history["warehouse_id"] == warehouse_id)
        ]
        grouped = peer_demand.groupby("sku_id")["units"]
        horizons: Dict[str, Dict[str, float]] = {}
        for horizon in HORIZON_STEPS:
            recent = grouped.tail(horizon).groupby(peer_demand["sku_id"]).sum() if not peer_demand.empty else pd.Series(dtype=float)
            values = recent.values if len(recent.values) else np.array([float(target["base_demand"]) * horizon / 7.0])
            horizons[f"{horizon}d"] = {
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }
        return ForecastOutput(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            horizons=horizons,
            explanation=ModelExplanation(
                engine=self.engine_name,
                summary="Cold-start forecast transferred from the nearest SKU cohort.",
                confidence=0.58,
                feature_contributions={"category_match": 0.4, "zone_match": 0.3, "cost_similarity": 0.2, "lead_time_similarity": 0.1},
                evidence=peer_ids,
            ),
            model_name="similarity_transfer",
            drift_detected=False,
        )

    def predict(
        self,
        sku_id: str,
        warehouse_id: str,
        demand_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
    ) -> ForecastOutput:
        frame, past_cols, future_cols, static_cols = self.build_feature_frame(demand_history, sku_catalog)
        
        # Align features if model is loaded and preprocessing has column info
        if self.model is not None and "past_cols" in self.preprocessing:
            trained_past = self.preprocessing["past_cols"]
            trained_future = self.preprocessing["future_cols"]
            trained_static = self.preprocessing["static_cols"]
            
            # Ensure all columns exist, fill with 0
            for col in set(trained_past + trained_future + trained_static):
                if col not in frame.columns:
                    frame[col] = 0.0
            
            past_cols = trained_past
            future_cols = trained_future
            static_cols = trained_static

        group = frame[
            (frame["sku_id"] == sku_id) & (frame["warehouse_id"] == warehouse_id)
        ].sort_values("timestamp")
        if self.model is None:
            raise RuntimeError("train the model before inference")
        if len(group) < self.preprocessing["sequence_length"] + max(HORIZON_STEPS):
            return self.cold_start_forecast(sku_id, warehouse_id, demand_history, sku_catalog)

        baseline = group["units"].tail(90).to_numpy(dtype=float)
        recent = group["units"].tail(30).to_numpy(dtype=float)
        drift = self.detector.evaluate(baseline, recent if len(recent) >= 10 else baseline)
        past, future, static = self._prepare_single_sample(
            frame, past_cols, future_cols, static_cols, sku_id, warehouse_id
        )
        self.model.eval()
        with torch.no_grad():
            prediction, importance = self.model(past, future, static)
        prediction_np = torch.expm1(prediction.squeeze(0)).cpu().numpy()
        importance_np = importance.squeeze(0).cpu().numpy()
        feature_map = {
            name: float(score)
            for name, score in zip(past_cols, importance_np)
        }
        horizons = {}
        for idx, horizon in enumerate(HORIZON_STEPS):
            horizons[f"{horizon}d"] = {
                "p10": float(max(0.0, prediction_np[idx, 0])),
                "p50": float(max(0.0, prediction_np[idx, 1])),
                "p90": float(max(0.0, prediction_np[idx, 2])),
            }
        return ForecastOutput(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            horizons=horizons,
            explanation=ModelExplanation(
                engine=self.engine_name,
                summary=(
                    "Forecast combines temporal attention over recent demand, known future signals, and static SKU attributes."
                ),
                confidence=float(np.clip(1.0 - drift.psi, 0.2, 0.95)),
                feature_contributions=feature_map,
                evidence=[
                    f"model={self.model_name}",
                    f"psi={drift.psi:.4f}",
                    f"kl={drift.kl_divergence:.4f}",
                ],
            ),
            model_name=self.model_name,
            drift_detected=drift.should_retrain,
        )
