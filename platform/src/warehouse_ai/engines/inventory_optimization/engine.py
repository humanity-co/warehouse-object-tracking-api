from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from warehouse_ai.core.schemas import DecisionAction, EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult
from warehouse_ai.utils.rl import GaussianActorCritic, TrajectoryBatch, compute_gae, ppo_update


def service_level_to_z(service_level: float) -> float:
    levels = np.array([0.90, 0.95, 0.97, 0.98, 0.99])
    z_scores = np.array([1.28, 1.64, 1.88, 2.05, 2.33])
    return float(np.interp(service_level, levels, z_scores))


@dataclass
class InventoryPlan:
    sku_id: str
    reorder_quantity: int
    reorder_eta_days: int
    transfer_quantity: int
    transfer_route: str
    storage_zone_bias: float
    safety_stock_by_warehouse: Dict[str, int]
    supplier_reliability: float
    stock_projection: List[Dict[str, Any]]
    cost_breakdown: Dict[str, float]
    explanation: ModelExplanation
    source_model: str


class InventoryControlEnvironment:
    """Multi-warehouse sequential inventory simulator."""

    def __init__(
        self,
        sku_id: str,
        demand_slice: pd.DataFrame,
        inventory_slice: pd.DataFrame,
        catalog_row: pd.Series,
        warehouse_capacities: Dict[str, int],
        episode_horizon: int = 45,
    ) -> None:
        self.sku_id = sku_id
        self.catalog_row = catalog_row
        self.episode_horizon = episode_horizon
        self.warehouse_ids = sorted(demand_slice["warehouse_id"].unique().tolist())
        self.capacity = np.array([warehouse_capacities.get(warehouse_id, 40_000) for warehouse_id in self.warehouse_ids], dtype=np.float32)
        demand_pivot = (
            demand_slice.pivot(index="timestamp", columns="warehouse_id", values="units").fillna(0.0).sort_index()
        )
        inventory_pivot = (
            inventory_slice.pivot(index="timestamp", columns="warehouse_id", values="on_hand").ffill().fillna(0.0).sort_index()
        )
        self.timeline = demand_pivot.index.tolist()
        self.demand = demand_pivot.reindex(columns=self.warehouse_ids).to_numpy(dtype=np.float32)
        self.inventory_history = inventory_pivot.reindex(columns=self.warehouse_ids).to_numpy(dtype=np.float32)
        self.lead_time = int(catalog_row["lead_time_days"])
        self.service_level = float(catalog_row["service_level_target"])
        self.holding_cost = float(catalog_row["unit_cost"]) * 0.015
        self.stockout_penalty = float(catalog_row["unit_cost"]) * 2.4
        self.transfer_cost = float(catalog_row["unit_cost"]) * 0.3
        self.waste_cost = float(catalog_row["unit_cost"]) * (2.0 / max(float(catalog_row["shelf_life_days"]), 1.0))
        self.supplier_reliability = float(np.clip(0.82 + 0.015 * (7 - hash(catalog_row["supplier_id"]) % 7), 0.7, 0.98))
        self.reset()

    @property
    def state_dim(self) -> int:
        return len(self.warehouse_ids) * 5 + 5

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self, start_index: Optional[int] = None) -> np.ndarray:
        max_start = max(len(self.timeline) - self.episode_horizon - 1, 1)
        if start_index is None:
            self.start_index = int(np.random.randint(0, max_start))
        else:
            self.start_index = start_index
        self.t = 0
        self.current_index = self.start_index
        self.inventory = self.inventory_history[self.current_index].astype(np.float32).copy()
        self.pending = [[] for _ in self.warehouse_ids]
        self.total_cost = 0.0
        self.fulfilled_units = 0.0
        self.total_demand_units = 0.0
        return self._state()

    def _forecast_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        start = max(0, self.current_index - 14)
        history = self.demand[start : self.current_index + 1]
        mean = history.mean(axis=0)
        std = history.std(axis=0) + 1e-3
        return mean, std

    def _safety_stock(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        z = service_level_to_z(self.service_level)
        return mean * self.lead_time + z * std * np.sqrt(max(self.lead_time, 1))

    def _state(self) -> np.ndarray:
        forecast_mean, forecast_std = self._forecast_stats()
        safety_stock = self._safety_stock(forecast_mean, forecast_std)
        cover = self.inventory / np.maximum(forecast_mean + 1e-3, 1.0)
        capacity_util = self.inventory / np.maximum(self.capacity, 1.0)
        pending_qty = np.array([sum(qty for _, qty in queue) for queue in self.pending], dtype=np.float32)
        scalar = np.array(
            [
                self.supplier_reliability,
                float(self.lead_time),
                float(self.holding_cost),
                float(self.stockout_penalty),
                float(self.waste_cost),
            ],
            dtype=np.float32,
        )
        return np.concatenate([cover, forecast_mean, forecast_std, capacity_util, pending_qty, scalar]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reorder_ratio = float((action[0] + 1.0) / 2.0)
        expedite_ratio = float((action[1] + 1.0) / 2.0)
        transfer_ratio = float(action[2])
        slot_bias = float((action[3] + 1.0) / 2.0)
        for idx, queue in enumerate(self.pending):
            arrivals = [qty for eta, qty in queue if eta <= self.t]
            if arrivals:
                self.inventory[idx] += sum(arrivals)
            self.pending[idx] = [(eta, qty) for eta, qty in queue if eta > self.t]

        forecast_mean, forecast_std = self._forecast_stats()
        safety_stock = self._safety_stock(forecast_mean, forecast_std)
        excess = self.inventory - safety_stock
        donor_idx = int(np.argmax(excess))
        receiver_idx = int(np.argmin(excess))
        transfer_qty = 0.0
        if donor_idx != receiver_idx and abs(transfer_ratio) > 0.05:
            direction = 1 if transfer_ratio > 0 else -1
            src_idx = donor_idx if direction > 0 else receiver_idx
            dst_idx = receiver_idx if direction > 0 else donor_idx
            transferable = max(0.0, self.inventory[src_idx] - safety_stock[src_idx])
            transfer_qty = float(abs(transfer_ratio) * transferable)
            self.inventory[src_idx] -= transfer_qty
            self.inventory[dst_idx] += transfer_qty

        gap = np.maximum(0.0, safety_stock + forecast_mean - self.inventory)
        total_gap = float(gap.sum())
        reorder_qty = reorder_ratio * total_gap
        allocation = gap / max(total_gap, 1.0)
        effective_lead = max(1, int(round(self.lead_time * (1.0 - 0.45 * expedite_ratio))))
        expected_fill = self.supplier_reliability * (0.82 + 0.18 * expedite_ratio)
        for idx, share in enumerate(allocation):
            qty = float(reorder_qty * share * expected_fill)
            if qty > 0:
                self.pending[idx].append((self.t + effective_lead, qty))

        demand = self.demand[self.current_index]
        fulfilled = np.minimum(self.inventory, demand)
        shortages = np.maximum(0.0, demand - self.inventory)
        self.inventory = np.maximum(0.0, self.inventory - demand)
        self.total_demand_units += float(demand.sum())
        self.fulfilled_units += float(fulfilled.sum())

        holding = float((self.inventory * self.holding_cost).sum())
        shortage_cost = float((shortages * self.stockout_penalty).sum())
        reorder_cost = reorder_qty * float(self.catalog_row["unit_cost"]) * (1.0 + 0.25 * expedite_ratio)
        transfer_cost = transfer_qty * self.transfer_cost
        waste_cost = float((np.maximum(0.0, self.inventory - safety_stock * 1.6) * self.waste_cost).sum())
        slotting_reward = float(slot_bias * fulfilled.sum() * 0.02)
        capacity_penalty = float(np.maximum(0.0, self.inventory - self.capacity * 0.95).sum() * 0.1)
        cost = holding + shortage_cost + reorder_cost + transfer_cost + waste_cost + capacity_penalty - slotting_reward
        self.total_cost += cost

        self.t += 1
        self.current_index += 1
        done = self.t >= self.episode_horizon or self.current_index >= len(self.timeline) - 1
        info = {
            "demand": demand,
            "fulfilled": fulfilled,
            "shortages": shortages,
            "reorder_qty": reorder_qty,
            "transfer_qty": transfer_qty,
            "effective_lead": effective_lead,
            "slot_bias": slot_bias,
            "safety_stock": safety_stock,
            "service_level": float(self.fulfilled_units / max(self.total_demand_units, 1.0)),
            "cost": cost,
        }
        return self._state(), -cost, done, info


class InventoryOptimizationEngine(BaseEngine):
    engine_name = "inventory_optimization"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.policy: Optional[GaussianActorCritic] = None
        self.training_summary: Dict[str, Any] = {}

    def build_environments(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        warehouses: pd.DataFrame,
    ) -> List[InventoryControlEnvironment]:
        warehouse_capacities = warehouses.set_index("warehouse_id")["capacity_units"].to_dict()
        envs: List[InventoryControlEnvironment] = []
        for _, catalog_row in sku_catalog.head(12).iterrows():
            sku_id = catalog_row["sku_id"]
            demand_slice = demand_history[demand_history["sku_id"] == sku_id]
            inventory_slice = inventory_history[inventory_history["sku_id"] == sku_id]
            if len(demand_slice) < 80 or demand_slice["warehouse_id"].nunique() < 2:
                continue
            envs.append(
                InventoryControlEnvironment(
                    sku_id=sku_id,
                    demand_slice=demand_slice,
                    inventory_slice=inventory_slice,
                    catalog_row=catalog_row,
                    warehouse_capacities=warehouse_capacities,
                )
            )
        if not envs:
            raise ValueError("insufficient data to build inventory environments")
        return envs

    def train(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        warehouses: pd.DataFrame,
        episodes: int = 24,
        learning_rate: float = 3e-4,
    ) -> TrainResult:
        envs = self.build_environments(demand_history, inventory_history, sku_catalog, warehouses)
        sample_env = envs[0]
        self.policy = GaussianActorCritic(sample_env.state_dim, sample_env.action_dim, hidden_dim=128)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        episode_rewards: List[float] = []
        service_levels: List[float] = []
        for _ in range(episodes):
            env = envs[np.random.randint(0, len(envs))]
            state = env.reset()
            states: List[torch.Tensor] = []
            actions: List[torch.Tensor] = []
            log_probs: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            dones: List[torch.Tensor] = []
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = self.policy.sample(state_tensor)
                next_state, reward, done, info = env.step(action.squeeze(0).detach().numpy())
                states.append(state_tensor.squeeze(0))
                actions.append(action.squeeze(0).detach())
                log_probs.append(log_prob.squeeze(0).detach())
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                values.append(value.squeeze(0).detach())
                dones.append(torch.tensor(float(done), dtype=torch.float32))
                state = next_state
            reward_tensor = torch.stack(rewards)
            value_tensor = torch.stack(values)
            done_tensor = torch.stack(dones)
            returns, advantages = compute_gae(reward_tensor, value_tensor, done_tensor)
            batch = TrajectoryBatch(
                states=torch.stack(states),
                actions=torch.stack(actions),
                old_log_probs=torch.stack(log_probs),
                returns=returns.detach(),
                advantages=advantages.detach(),
            )
            ppo_update(self.policy, optimizer, batch)
            episode_rewards.append(float(reward_tensor.sum().item()))
            service_levels.append(float(info["service_level"]))
        self.training_summary = {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "mean_service_level": float(np.mean(service_levels)),
            "episodes": episodes,
        }
        artifact_path = self.save_state(
            self.policy,
            self.artifacts_dir / f"{self.engine_name}_ppo.pt",
            extra={"training_summary": self.training_summary},
        )
        metrics = [
            EvaluationMetric(name="mean_episode_reward", value=self.training_summary["mean_episode_reward"]),
            EvaluationMetric(name="mean_service_level", value=self.training_summary["mean_service_level"]),
        ]
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=metrics,
            metadata={"policy": "ppo"},
        )

    def _simulate_action_cost(
        self,
        env: InventoryControlEnvironment,
        action: np.ndarray,
        forecast_mean: np.ndarray,
        forecast_std: np.ndarray,
        simulations: int = 32,
    ) -> float:
        z = service_level_to_z(env.service_level)
        safety = forecast_mean * env.lead_time + z * forecast_std * np.sqrt(max(env.lead_time, 1))
        donor_idx = int(np.argmax(env.inventory - safety))
        receiver_idx = int(np.argmin(env.inventory - safety))
        reorder_ratio = float((action[0] + 1.0) / 2.0)
        expedite_ratio = float((action[1] + 1.0) / 2.0)
        transfer_ratio = float(action[2])
        slot_bias = float((action[3] + 1.0) / 2.0)
        total_gap = float(np.maximum(0.0, safety + forecast_mean - env.inventory).sum())
        reorder_qty = reorder_ratio * total_gap
        transfer_qty = abs(transfer_ratio) * max(0.0, env.inventory[donor_idx] - safety[donor_idx])
        demand_samples = np.maximum(
            0.0,
            np.random.normal(
                loc=forecast_mean,
                scale=np.maximum(forecast_std, 1.0),
                size=(simulations, len(env.warehouse_ids)),
            ),
        )
        remaining = np.maximum(0.0, env.inventory + reorder_qty / len(env.warehouse_ids) - demand_samples)
        shortages = np.maximum(0.0, demand_samples - (env.inventory + reorder_qty / len(env.warehouse_ids)))
        expected_cost = (
            (remaining * env.holding_cost).sum(axis=1)
            + (shortages * env.stockout_penalty).sum(axis=1)
            + reorder_qty * float(env.catalog_row["unit_cost"]) * (1.0 + 0.25 * expedite_ratio)
            + transfer_qty * env.transfer_cost
            - slot_bias * demand_samples.sum(axis=1) * 0.02
        )
        if donor_idx == receiver_idx:
            transfer_qty = 0.0
        return float(expected_cost.mean())

    def fallback_dynamic_programming(self, env: InventoryControlEnvironment) -> np.ndarray:
        forecast_mean, forecast_std = env._forecast_stats()
        action_grid: List[np.ndarray] = []
        for reorder in np.linspace(-1.0, 1.0, 5):
            for expedite in np.linspace(-1.0, 1.0, 3):
                for transfer in np.linspace(-0.8, 0.8, 5):
                    for slot_bias in np.linspace(-0.5, 0.5, 3):
                        action_grid.append(np.array([reorder, expedite, transfer, slot_bias], dtype=np.float32))
        scored = [
            (self._simulate_action_cost(env, action, forecast_mean, forecast_std), action)
            for action in action_grid
        ]
        return min(scored, key=lambda item: item[0])[1]

    def recommend(
        self,
        sku_id: str,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        warehouses: pd.DataFrame,
    ) -> InventoryPlan:
        envs = self.build_environments(demand_history, inventory_history, sku_catalog, warehouses)
        target_env = next((env for env in envs if env.sku_id == sku_id), None)
        if target_env is None:
            raise KeyError("SKU not available for inventory planning")
        state = target_env.reset(start_index=max(len(target_env.timeline) - target_env.episode_horizon - 2, 0))
        forecast_mean, forecast_std = target_env._forecast_stats()
        safety_stock = target_env._safety_stock(forecast_mean, forecast_std)
        if self.policy is None:
            action_np = self.fallback_dynamic_programming(target_env)
            source_model = "stochastic_dynamic_programming"
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action_np = action.squeeze(0).numpy()
            source_model = "ppo"
        _, _, _, info = target_env.step(action_np)
        
        reorder_qty = int(round(float(info["reorder_qty"])))
        transfer_qty = int(round(float(info["transfer_qty"])))
        storage_zone_bias = float(info["slot_bias"])
        
        donor_idx = int(np.argmax(target_env.inventory - safety_stock))
        receiver_idx = int(np.argmin(target_env.inventory - safety_stock))
        transfer_route = f"{target_env.warehouse_ids[donor_idx]}->{target_env.warehouse_ids[receiver_idx]}"

        explanation = ModelExplanation(
            engine=self.engine_name,
            summary=(
                "Recommendation minimizes holding, shortage, transport, and waste costs while respecting safety stock and supplier reliability."
            ),
            confidence=float(np.clip(target_env.supplier_reliability - (0.2 if source_model != "ppo" else 0.05), 0.4, 0.95)),
            feature_contributions={
                "forecast_mean": float(forecast_mean.mean()),
                "forecast_volatility": float(forecast_std.mean()),
                "supplier_reliability": float(target_env.supplier_reliability),
                "inventory_gap": float(np.maximum(0.0, safety_stock + forecast_mean - target_env.inventory).sum()),
            },
            evidence=[
                f"effective_lead={info['effective_lead']}",
                f"service_level={info['service_level']:.3f}",
                f"source_model={source_model}",
            ],
        )

        # 14-Day Projection Simulation
        projection: List[Dict[str, Any]] = []
        sim_inventory = target_env.inventory.copy()
        for day in range(14):
            day_demand = forecast_mean * (1.1 if day < 3 else 1.0) # Slight trend
            sim_inventory = np.maximum(0.0, sim_inventory - day_demand)
            if day == info["effective_lead"]:
                sim_inventory += reorder_qty / len(target_env.warehouse_ids)
            
            projection.append({
                "day": day,
                "on_hand": float(sim_inventory.sum()),
                "safety_stock": float(safety_stock.sum()),
                "reorder_point": float((safety_stock + forecast_mean).sum())
            })

        cost_breakdown = {
            "holding": float((target_env.inventory * target_env.holding_cost).sum()),
            "shortage": float(info["shortages"].sum() * target_env.stockout_penalty),
            "transport": float(info["transfer_qty"] * target_env.transfer_cost),
            "opportunity": float(info["slot_bias"] * 50.0)
        }

        return InventoryPlan(
            sku_id=sku_id,
            reorder_quantity=reorder_qty,
            reorder_eta_days=int(info["effective_lead"]),
            transfer_quantity=transfer_qty,
            transfer_route=transfer_route,
            storage_zone_bias=storage_zone_bias,
            safety_stock_by_warehouse={
                warehouse_id: int(round(stock))
                for warehouse_id, stock in zip(target_env.warehouse_ids, safety_stock)
            },
            supplier_reliability=float(target_env.supplier_reliability),
            stock_projection=projection,
            cost_breakdown=cost_breakdown,
            explanation=explanation,
            source_model=source_model,
        )

    def as_action(self, plan: InventoryPlan) -> DecisionAction:
        return DecisionAction(
            action_type="inventory_replenishment",
            entity_id=plan.sku_id,
            recommended_value=plan.reorder_quantity,
            confidence=plan.explanation.confidence,
            explanation=plan.explanation,
            created_at=pd.Timestamp.utcnow().to_pydatetime(),
        )
