from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult
from warehouse_ai.utils.rl import CategoricalActorCritic, TrajectoryBatch, compute_gae, ppo_update


@dataclass
class SlottingPlan:
    assignments: Dict[str, str]
    objective_score: float
    congestion_score: float
    constraint_violations: int
    explanation: ModelExplanation


class SlottingEnvironment:
    def __init__(self, item_features: np.ndarray, zone_features: np.ndarray, compatibility: np.ndarray) -> None:
        self.item_features = item_features.astype(np.float32)
        self.zone_features = zone_features.astype(np.float32)
        self.compatibility = compatibility.astype(np.float32)
        self.num_items = item_features.shape[0]
        self.num_zones = zone_features.shape[0]
        self.state_dim = self.item_features.shape[1] + self.num_zones
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.index = 0
        self.zone_load = np.zeros(self.num_zones, dtype=np.float32)
        return self._state(), self._mask()

    def _state(self) -> np.ndarray:
        item = self.item_features[self.index]
        load = self.zone_load / np.maximum(self.zone_load.sum(), 1.0)
        return np.concatenate([item, load], dtype=np.float32)

    def _mask(self) -> np.ndarray:
        return self.compatibility[self.index]

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[str, float]]:
        item = self.item_features[self.index]
        zone = self.zone_features[action]
        velocity = item[0]
        affinity = item[1]
        volume = item[2]
        weight = item[3]
        travel_score = zone[0]
        density = zone[1]
        congestion = zone[2] + self.zone_load[action] * 0.05
        constraint_penalty = 0.0
        if self.compatibility[self.index, action] < 0.5:
            constraint_penalty += 15.0
        if weight > zone[3]:
            constraint_penalty += 6.0
        if volume > zone[4]:
            constraint_penalty += 6.0
        reward = velocity * (1.5 - travel_score) + affinity * (1.2 - congestion) + density - constraint_penalty
        self.zone_load[action] += volume
        self.index += 1
        done = self.index >= self.num_items
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        next_mask = self._mask() if not done else np.ones(self.num_zones, dtype=np.float32)
        info = {
            "travel_score": float(travel_score),
            "congestion": float(congestion),
            "constraint_penalty": float(constraint_penalty),
        }
        return (next_state, next_mask), float(reward), done, info


class SmartSlottingEngine(BaseEngine):
    engine_name = "smart_slotting"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.velocity_model = GradientBoostingRegressor(random_state=42)
        self.policy: Optional[CategoricalActorCritic] = None
        self.zone_lookup: List[str] = []
        self.embedding_dim = 6

    def _build_affinity_matrix(self, order_lines: pd.DataFrame, sku_catalog: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        order_map = order_lines.groupby("order_id")["sku_id"].apply(list)
        sku_ids = sku_catalog["sku_id"].tolist()
        index = {sku_id: idx for idx, sku_id in enumerate(sku_ids)}
        matrix = np.zeros((len(sku_ids), len(sku_ids)), dtype=np.float32)
        for items in order_map:
            unique_items = list(dict.fromkeys(items))
            for i, left in enumerate(unique_items):
                for right in unique_items[i + 1 :]:
                    matrix[index[left], index[right]] += 1
                    matrix[index[right], index[left]] += 1
        return matrix, sku_ids

    def _abc_xyz(self, demand_history: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            demand_history.groupby("sku_id")
            .agg(mean_units=("units", "mean"), std_units=("units", "std"), total_units=("units", "sum"))
            .fillna(0.0)
        )
        grouped["cv"] = grouped["std_units"] / np.maximum(grouped["mean_units"], 1.0)
        grouped["abc"] = pd.qcut(grouped["total_units"].rank(method="first"), 3, labels=["C", "B", "A"])
        grouped["xyz"] = pd.cut(grouped["cv"], bins=[-1, 0.4, 0.9, np.inf], labels=["X", "Y", "Z"])
        return grouped.reset_index().rename(
            columns={
                "mean_units": "global_mean_units",
                "std_units": "global_std_units",
                "cv": "global_cv",
            }
        )

    def _prepare_items(
        self,
        demand_history: pd.DataFrame,
        order_lines: pd.DataFrame,
        sku_catalog: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        affinity_matrix, sku_ids = self._build_affinity_matrix(order_lines, sku_catalog)
        svd = TruncatedSVD(n_components=min(self.embedding_dim, max(2, affinity_matrix.shape[0] - 1)), random_state=42)
        embeddings = svd.fit_transform(affinity_matrix + np.eye(len(sku_ids), dtype=np.float32))
        abc_xyz = self._abc_xyz(demand_history)
        velocity_frame = (
            demand_history.groupby(["sku_id", "warehouse_id"])
            .agg(
                mean_units=("units", "mean"),
                std_units=("units", "std"),
                promo_rate=("promotion_flag", "mean"),
                external_signal=("external_signal", "mean"),
            )
            .fillna(0.0)
            .reset_index()
        )
        features = velocity_frame.merge(abc_xyz, on="sku_id", how="left").merge(sku_catalog, on="sku_id", how="left")
        encoded = pd.get_dummies(features[["warehouse_id", "category", "storage_zone", "abc", "xyz"]], dtype=float)
        velocity_inputs = pd.concat(
            [
                features[["std_units", "promo_rate", "external_signal", "weight_kg", "length_cm", "width_cm", "height_cm"]].reset_index(drop=True),
                encoded.reset_index(drop=True),
            ],
            axis=1,
        )
        self.velocity_model.fit(velocity_inputs, features["mean_units"])
        features["predicted_velocity"] = self.velocity_model.predict(velocity_inputs)
        embedding_frame = pd.DataFrame(embeddings, columns=[f"emb_{idx}" for idx in range(embeddings.shape[1])])
        embedding_frame["sku_id"] = sku_ids
        merged = features.merge(embedding_frame, on="sku_id", how="left")
        return merged, embeddings

    def _prepare_zones(self, route_nodes: pd.DataFrame) -> pd.DataFrame:
        zones = (
            route_nodes.groupby(["warehouse_id", "zone_id"])
            .agg(
                avg_x=("x", "mean"),
                avg_y=("y", "mean"),
                pick_face_share=("pick_face", "mean"),
                slots=("node_id", "count"),
            )
            .reset_index()
        )
        zones["travel_score"] = np.sqrt(zones["avg_x"] ** 2 + zones["avg_y"] ** 2) / np.sqrt(
            route_nodes["x"].max() ** 2 + route_nodes["y"].max() ** 2
        )
        zones["density_score"] = zones["slots"] / np.maximum(zones["slots"].max(), 1)
        zones["congestion_score"] = 1.0 - zones["pick_face_share"]
        zones["weight_limit"] = np.where(zones["pick_face_share"] > 0.5, 8.0, 20.0)
        zones["volume_limit"] = zones["slots"] * 5000.0
        self.zone_lookup = zones["zone_id"].tolist()
        return zones

    def _compatibility_matrix(self, items: pd.DataFrame, zones: pd.DataFrame) -> np.ndarray:
        compatibility = np.ones((len(items), len(zones)), dtype=np.float32)
        for item_idx, item in items.reset_index(drop=True).iterrows():
            for zone_idx, zone in zones.reset_index(drop=True).iterrows():
                if item["warehouse_id"] != zone["warehouse_id"]:
                    compatibility[item_idx, zone_idx] = 0.0
                if item["storage_zone"] == "chilled" and "Z1" not in zone["zone_id"]:
                    compatibility[item_idx, zone_idx] *= 0.6
                if item["weight_kg"] > zone["weight_limit"]:
                    compatibility[item_idx, zone_idx] *= 0.3
        return compatibility

    def _seed_assignments(self, items: pd.DataFrame, zones: pd.DataFrame) -> np.ndarray:
        features = items[["predicted_velocity", "mean_units", "weight_kg", "length_cm"]].fillna(0.0).to_numpy()
        clusters = min(len(zones), max(2, len(items) // 4))
        labels = KMeans(n_clusters=clusters, random_state=42, n_init="auto").fit_predict(features)
        zone_order = zones.sort_values(["travel_score", "congestion_score"]).reset_index(drop=True)
        cluster_to_zone = {cluster: idx % len(zone_order) for idx, cluster in enumerate(sorted(set(labels)))}
        assignments = np.array([cluster_to_zone[label] for label in labels], dtype=np.int64)
        return assignments

    def train(
        self,
        demand_history: pd.DataFrame,
        order_lines: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        route_nodes: pd.DataFrame,
        episodes: int = 30,
    ) -> TrainResult:
        items, _ = self._prepare_items(demand_history, order_lines, sku_catalog)
        zones = self._prepare_zones(route_nodes)
        compatibility = self._compatibility_matrix(items, zones)
        item_features = items[
            ["predicted_velocity", "mean_units", "length_cm", "weight_kg", "emb_0", "emb_1"]
        ].fillna(0.0).to_numpy(dtype=np.float32)
        zone_features = zones[
            ["travel_score", "density_score", "congestion_score", "weight_limit", "volume_limit"]
        ].to_numpy(dtype=np.float32)
        env = SlottingEnvironment(item_features, zone_features, compatibility)
        state_dim = item_features.shape[1] + len(zones)
        self.policy = CategoricalActorCritic(state_dim=state_dim, action_dim=len(zones), hidden_dim=128)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        rewards_history: List[float] = []
        seed_assignments = self._seed_assignments(items, zones)
        for _ in range(episodes):
            state, mask = env.reset()
            states: List[torch.Tensor] = []
            actions: List[torch.Tensor] = []
            log_probs: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            dones: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            done = False
            step = 0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                if np.random.rand() < 0.2:
                    action_idx = int(seed_assignments[min(step, len(seed_assignments) - 1)])
                    action = torch.tensor([action_idx], dtype=torch.int64)
                    with torch.no_grad():
                        logits, value = self.policy.forward(state_tensor, action_mask=mask_tensor)
                        log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)
                else:
                    action, log_prob, value = self.policy.sample(state_tensor, action_mask=mask_tensor)
                (state, mask), reward, done, _ = env.step(int(action.item()))
                states.append(state_tensor.squeeze(0))
                actions.append(action.squeeze(0).detach())
                log_probs.append(log_prob.squeeze(0).detach())
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                values.append(value.squeeze(0).detach())
                dones.append(torch.tensor(float(done), dtype=torch.float32))
                masks.append(mask_tensor.squeeze(0))
                step += 1
            returns, advantages = compute_gae(torch.stack(rewards), torch.stack(values), torch.stack(dones))
            batch = TrajectoryBatch(
                states=torch.stack(states),
                actions=torch.stack(actions),
                old_log_probs=torch.stack(log_probs),
                returns=returns.detach(),
                advantages=advantages.detach(),
                action_mask=torch.stack(masks),
            )
            ppo_update(self.policy, optimizer, batch)
            rewards_history.append(float(torch.stack(rewards).sum().item()))
        artifact_path = self.save_state(
            self.policy,
            self.artifacts_dir / f"{self.engine_name}_policy.pt",
            extra={"zones": self.zone_lookup},
        )
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=[
                EvaluationMetric(name="mean_reward", value=float(np.mean(rewards_history))),
                EvaluationMetric(name="zones", value=float(len(zones))),
            ],
            metadata={"optimization": "clustering + ppo"},
        )

    def optimize(
        self,
        demand_history: pd.DataFrame,
        order_lines: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        route_nodes: pd.DataFrame,
    ) -> SlottingPlan:
        if self.policy is None:
            raise RuntimeError("train the slotting policy before optimization")
        items, _ = self._prepare_items(demand_history, order_lines, sku_catalog)
        zones = self._prepare_zones(route_nodes)
        compatibility = self._compatibility_matrix(items, zones)
        item_features = items[
            ["predicted_velocity", "mean_units", "length_cm", "weight_kg", "emb_0", "emb_1"]
        ].fillna(0.0).to_numpy(dtype=np.float32)
        zone_features = zones[
            ["travel_score", "density_score", "congestion_score", "weight_limit", "volume_limit"]
        ].to_numpy(dtype=np.float32)
        env = SlottingEnvironment(item_features, zone_features, compatibility)
        state, mask = env.reset()
        assignments: Dict[str, str] = {}
        total_reward = 0.0
        congestion_scores: List[float] = []
        constraint_violations = 0
        item_ids = items["sku_id"].tolist()
        index = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.policy.sample(state_tensor, action_mask=mask_tensor)
            action_idx = int(action.item())
            assignments[item_ids[index]] = self.zone_lookup[action_idx]
            (state, mask), reward, done, info = env.step(action_idx)
            total_reward += reward
            congestion_scores.append(info["congestion"])
            if info["constraint_penalty"] > 0:
                constraint_violations += 1
            index += 1
        return SlottingPlan(
            assignments=assignments,
            objective_score=float(total_reward / max(len(assignments), 1)),
            congestion_score=float(np.mean(congestion_scores) if congestion_scores else 0.0),
            constraint_violations=constraint_violations,
            explanation=ModelExplanation(
                engine=self.engine_name,
                summary="Slot assignments were seeded from affinity and velocity clusters, then refined with RL under storage constraints.",
                confidence=float(np.clip(0.88 - constraint_violations / max(len(assignments), 1), 0.35, 0.95)),
                feature_contributions={
                    "predicted_velocity": float(items["predicted_velocity"].mean()),
                    "travel_score": float(zones["travel_score"].mean()),
                    "congestion_score": float(np.mean(congestion_scores) if congestion_scores else 0.0),
                },
                evidence=[
                    f"assigned_items={len(assignments)}",
                    f"constraint_violations={constraint_violations}",
                ],
            ),
        )
