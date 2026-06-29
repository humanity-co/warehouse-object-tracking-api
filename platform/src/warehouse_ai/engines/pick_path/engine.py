from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn

from warehouse_ai.core.schemas import EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult
from warehouse_ai.utils.rl import TrajectoryBatch, compute_gae, ppo_update


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


class GraphRoutingPolicy(nn.Module):
    def __init__(self, max_nodes: int, node_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.logit_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_flat = states[:, : self.max_nodes * self.node_dim]
        current_one_hot = states[:, self.max_nodes * self.node_dim :]
        nodes = node_flat.view(-1, self.max_nodes, self.node_dim)
        hidden = torch.relu(self.node_encoder(nodes))
        coords = nodes[:, :, :2]
        dist = torch.cdist(coords, coords, p=2)
        adjacency = torch.exp(-dist)
        adjacency = adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-6)
        for _ in range(2):
            hidden = torch.relu(self.self_proj(hidden) + self.neighbor_proj(torch.bmm(adjacency, hidden)))
        current_context = torch.sum(hidden * current_one_hot.unsqueeze(-1), dim=1)
        global_context = hidden.mean(dim=1)
        return hidden, torch.cat([current_context, global_context], dim=-1)

    def forward(self, states: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, context = self._encode(states)
        repeated_context = context.unsqueeze(1).expand(-1, self.max_nodes, -1)
        logits = self.logit_head(torch.cat([hidden, repeated_context], dim=-1)).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.value_head(context).squeeze(-1)
        return logits, value

    def sample(self, states: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(states, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(states, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


@dataclass
class RoutePlan:
    warehouse_id: str
    route_nodes: List[str]
    estimated_distance: float
    wave_id: str
    clustered_orders: List[str]
    explanation: ModelExplanation
    source_model: str
    safety_score: float = 0.0
    speed_index: float = 0.0


class RoutingEnvironment:
    def __init__(self, graph: nx.Graph, node_features: np.ndarray, node_ids: List[str]) -> None:
        self.graph = graph
        self.node_features = node_features.astype(np.float32)
        self.node_ids = node_ids
        self.max_nodes = len(node_ids)
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.current_index = 0
        self.visited = np.zeros(self.max_nodes, dtype=np.float32)
        self.visited[0] = 1.0
        self.current_node = 0
        return self._state(), self._mask()

    def _state(self) -> np.ndarray:
        return np.concatenate([self.node_features.flatten(), np.eye(self.max_nodes, dtype=np.float32)[self.current_node]])

    def _mask(self) -> np.ndarray:
        mask = 1.0 - self.visited
        mask[0] = 0.0
        return mask.astype(np.float32)

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[str, float]]:
        next_node = action
        current_id = self.node_ids[self.current_node]
        next_id = self.node_ids[next_node]
        distance = nx.shortest_path_length(self.graph, current_id, next_id, weight="weight")
        congestion = self.graph.nodes[next_id]["congestion"]
        reward = -(distance * (1.0 + congestion))
        self.current_node = next_node
        self.visited[next_node] = 1.0
        done = bool(self.visited[1:].all())
        if done:
            distance_back = nx.shortest_path_length(
                self.graph,
                self.node_ids[self.current_node],
                self.node_ids[0],
                weight="weight",
            )
            reward -= distance_back
        info = {"distance": float(distance), "congestion": float(congestion)}
        next_state = self._state()
        next_mask = self._mask()
        return (next_state, next_mask), float(reward), done, info


class PickPathOptimizationEngine(BaseEngine):
    engine_name = "pick_path_optimization"

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.policy: Optional[GraphRoutingPolicy] = None
        self.max_batch_nodes = 8

    def build_graph(self, route_nodes: pd.DataFrame) -> Dict[str, nx.Graph]:
        graphs: Dict[str, nx.Graph] = {}
        for warehouse_id, group in route_nodes.groupby("warehouse_id", sort=False):
            graph = nx.Graph()
            for _, row in group.iterrows():
                congestion = 0.1 + (0.3 if not row["pick_face"] else 0.0)
                graph.add_node(
                    row["node_id"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    zone_id=row["zone_id"],
                    congestion=congestion,
                )
            records = group.to_dict(orient="records")
            for left in records:
                for right in records:
                    if left["node_id"] == right["node_id"]:
                        continue
                    same_aisle = left["node_id"].split("-")[1] == right["node_id"].split("-")[1]
                    same_slot = left["node_id"].split("-")[2] == right["node_id"].split("-")[2]
                    if same_aisle or same_slot:
                        dist = euclidean((left["x"], left["y"]), (right["x"], right["y"]))
                        congestion = 1.0 + (graph.nodes[left["node_id"]]["congestion"] + graph.nodes[right["node_id"]]["congestion"]) / 2.0
                        graph.add_edge(left["node_id"], right["node_id"], weight=dist * congestion)
            graphs[warehouse_id] = graph
        return graphs

    def _assign_pick_nodes(self, order_lines: pd.DataFrame, route_nodes: pd.DataFrame) -> pd.DataFrame:
        nodes_by_wh = {
            warehouse_id: group["node_id"].tolist()
            for warehouse_id, group in route_nodes.groupby("warehouse_id", sort=False)
        }
        orders = order_lines.copy()
        orders["pick_node"] = orders.apply(
            lambda row: nodes_by_wh[row["warehouse_id"]][hash(row["sku_id"]) % len(nodes_by_wh[row["warehouse_id"]])],
            axis=1,
        )
        return orders

    def cluster_batches(self, order_lines: pd.DataFrame, route_nodes: pd.DataFrame) -> Dict[str, List[pd.DataFrame]]:
        routed_orders = self._assign_pick_nodes(order_lines, route_nodes)
        batches: Dict[str, List[pd.DataFrame]] = {}
        for warehouse_id, group in routed_orders.groupby("warehouse_id", sort=False):
            node_lookup = route_nodes[route_nodes["warehouse_id"] == warehouse_id].set_index("node_id")[["x", "y"]]
            coords = np.stack([node_lookup.loc[node_id].to_numpy(dtype=float) for node_id in group["pick_node"]])
            clusters = min(max(1, len(group) // self.max_batch_nodes + 1), 5)
            labels = KMeans(n_clusters=clusters, random_state=42, n_init="auto").fit_predict(coords)
            batches[warehouse_id] = [batch.reset_index(drop=True) for _, batch in group.groupby(labels)]
        return batches

    def astar_route(
        self,
        graph: nx.Graph,
        depot: str,
        pick_nodes: List[str],
        blocked_nodes: Optional[List[str]] = None,
    ) -> Tuple[List[str], float]:
        blocked = set(blocked_nodes or [])
        working_graph = graph.copy()
        working_graph.remove_nodes_from([node for node in blocked if node in working_graph])
        route = [depot]
        current = depot
        total_distance = 0.0
        remaining = [node for node in pick_nodes if node in working_graph]
        while remaining:
            candidates = []
            for node in remaining:
                heuristic = euclidean(
                    (working_graph.nodes[current]["x"], working_graph.nodes[current]["y"]),
                    (working_graph.nodes[node]["x"], working_graph.nodes[node]["y"]),
                )
                path = nx.astar_path(
                    working_graph,
                    current,
                    node,
                    heuristic=lambda left, right, g=working_graph: euclidean(
                        (g.nodes[left]["x"], g.nodes[left]["y"]),
                        (g.nodes[right]["x"], g.nodes[right]["y"]),
                    ),
                    weight="weight",
                )
                distance = nx.path_weight(working_graph, path, weight="weight")
                candidates.append((distance + heuristic, distance, node, path))
            _, distance, node, path = min(candidates, key=lambda item: item[0])
            route.extend(path[1:])
            total_distance += distance
            current = node
            remaining.remove(node)
        if depot in working_graph and current != depot:
            back_path = nx.astar_path(
                working_graph,
                current,
                depot,
                heuristic=lambda left, right, g=working_graph: euclidean(
                    (g.nodes[left]["x"], g.nodes[left]["y"]),
                    (g.nodes[right]["x"], g.nodes[right]["y"]),
                ),
                weight="weight",
            )
            route.extend(back_path[1:])
            total_distance += nx.path_weight(working_graph, back_path, weight="weight")
        return route, float(total_distance)

    def _routing_episode(self, graph: nx.Graph, batch: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        pick_nodes = list(dict.fromkeys(batch["pick_node"].tolist()))[: self.max_batch_nodes - 1]
        depot = sorted(graph.nodes())[0]
        episode_nodes = [depot] + pick_nodes
        if len(episode_nodes) < 2:
            raise ValueError("routing episode requires at least one pick node")
        padded_nodes = episode_nodes + [depot] * (self.max_batch_nodes - len(episode_nodes))
        features = []
        for idx, node_id in enumerate(padded_nodes):
            node = graph.nodes[node_id]
            features.append(
                [
                    node["x"] / 100.0,
                    node["y"] / 100.0,
                    node["congestion"],
                    0.0 if idx < len(episode_nodes) else 1.0,
                    1.0 if idx > 0 and idx < len(episode_nodes) else 0.0,
                ]
            )
        return np.array(features, dtype=np.float32), padded_nodes

    def train(
        self,
        order_lines: pd.DataFrame,
        route_nodes: pd.DataFrame,
        episodes: int = 28,
    ) -> TrainResult:
        graphs = self.build_graph(route_nodes)
        batches = self.cluster_batches(order_lines, route_nodes)
        node_dim = 5
        self.policy = GraphRoutingPolicy(max_nodes=self.max_batch_nodes, node_dim=node_dim, hidden_dim=96)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        reward_history: List[float] = []
        for _ in range(episodes):
            warehouse_id = sorted(batches.keys())[np.random.randint(0, len(batches))]
            batch_group = batches[warehouse_id][np.random.randint(0, len(batches[warehouse_id]))]
            node_features, node_ids = self._routing_episode(graphs[warehouse_id], batch_group)
            env = RoutingEnvironment(graphs[warehouse_id], node_features, node_ids)
            state, mask = env.reset()
            states: List[torch.Tensor] = []
            actions: List[torch.Tensor] = []
            log_probs: List[torch.Tensor] = []
            rewards: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            dones: List[torch.Tensor] = []
            masks: List[torch.Tensor] = []
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = self.policy.sample(state_tensor, action_mask=mask_tensor)
                (state, mask), reward, done, _ = env.step(int(action.item()))
                states.append(state_tensor.squeeze(0))
                actions.append(action.squeeze(0).detach())
                log_probs.append(log_prob.squeeze(0).detach())
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                values.append(value.squeeze(0).detach())
                dones.append(torch.tensor(float(done), dtype=torch.float32))
                masks.append(mask_tensor.squeeze(0))
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
            reward_history.append(float(torch.stack(rewards).sum().item()))
        artifact_path = self.save_state(
            self.policy,
            self.artifacts_dir / f"{self.engine_name}_gnn_policy.pt",
            extra={"max_batch_nodes": self.max_batch_nodes},
        )
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=[EvaluationMetric(name="mean_reward", value=float(np.mean(reward_history)))],
            metadata={"baseline": "astar", "advanced": "gnn+ppo"},
        )

    def plan(
        self,
        order_lines: pd.DataFrame,
        route_nodes: pd.DataFrame,
        blocked_nodes: Optional[List[str]] = None,
    ) -> List[RoutePlan]:
        graphs = self.build_graph(route_nodes)
        batches = self.cluster_batches(order_lines, route_nodes)
        plans: List[RoutePlan] = []
        for warehouse_id, grouped_batches in batches.items():
            graph = graphs[warehouse_id]
            for batch_index, batch in enumerate(grouped_batches):
                node_features, node_ids = self._routing_episode(graph, batch)
                pick_nodes = [node for node in node_ids[1:] if node != node_ids[0]]
                if self.policy is None:
                    route, distance = self.astar_route(graph, node_ids[0], pick_nodes, blocked_nodes=blocked_nodes)
                    source_model = "astar"
                else:
                    env = RoutingEnvironment(graph, node_features, node_ids)
                    state, mask = env.reset()
                    sequence = [node_ids[0]]
                    total_reward = 0.0
                    done = False
                    while not done:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            action, _, _ = self.policy.sample(state_tensor, action_mask=mask_tensor)
                        node_id = node_ids[int(action.item())]
                        if node_id not in sequence:
                            sequence.append(node_id)
                        (state, mask), reward, done, _ = env.step(int(action.item()))
                        total_reward += reward
                    route, distance = self.astar_route(graph, node_ids[0], sequence[1:], blocked_nodes=blocked_nodes)
                    source_model = "gnn_ppo"
                # Calculate premium metrics
                safety_score = float(np.clip(0.95 - (len(blocked_nodes or []) * 0.05), 0.5, 1.0))
                speed_index = float(np.clip(1.2 - (distance / 500.0), 0.4, 1.0))
                
                plans.append(
                    RoutePlan(
                        warehouse_id=warehouse_id,
                        route_nodes=route,
                        estimated_distance=distance,
                        wave_id=f"{warehouse_id}-wave-{batch_index+1:03d}",
                        clustered_orders=batch["order_id"].tolist(),
                        explanation=ModelExplanation(
                            engine=self.engine_name,
                            summary=f"Route minimizes distance with congestion-aware weights. | Safety: {safety_score:.2f} | Speed: {speed_index:.2f}",
                            confidence=float(np.clip(0.78 if source_model == "gnn_ppo" else 0.65, 0.4, 0.95)),
                            feature_contributions={
                                "batch_size": float(len(batch)),
                                "blocked_nodes": float(len(blocked_nodes or [])),
                                "estimated_distance": float(distance),
                                "safety_risk": 1.0 - safety_score,
                            },
                            evidence=[
                                f"warehouse={warehouse_id}",
                                f"source_model={source_model}",
                                f"safety={safety_score}",
                            ],
                        ),
                        source_model=source_model,
                        safety_score=safety_score,
                        speed_index=speed_index,
                    )
                )
        return plans
