#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# GRASP
# QUANT
# Dynamic Networks
# Link Prediction

# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs


"""
GRASP_Quant_Lotto_v2 — poboljšana varijanta.

v2: CSV samo Num1–Num7 (ne ceo red); RF n_jobs=1; Node2Vec workers=1; par→indeks dict;
    X_norm bez deljenja nulom; argparse; np.isclose za izjednačenje skorova u GRASP.
Grafika: `--plot` čuva PNG u `--out-dir`; prozor: `--show`.
torch 2.8.0 / dgl==1.1.3 — okruženje.

python3 GRASP_Quant_Lotto_v2.py --plot
python3 GRASP_Quant_Lotto_v2.py --plot --show

"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestRegressor
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ================= podrazumevano =================
SEED = 39
NODES = list(range(1, 40))
_DEFAULT_CSV = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4584_k23.csv")
RF_TREES = 300
TOP_EDGES_DEFAULT = 200
N2V_DIM = 32
N2V_WALK_LEN = 10
N2V_NUM_WALKS = 50
N2V_WINDOW = 5
ALPHA = 1.0
BETA = 1.5
GAMMA = 1.0
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "GRASP_Quant_Lotto_v2_out"


def maybe_plot_aggregate(
    g: nx.Graph,
    plot: bool,
    show_window: bool,
    out_dir: Path,
    layout_seed: int,
) -> None:
    if not plot:
        return
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=layout_seed)
    nx.draw_networkx_nodes(g, pos, node_size=40, alpha=0.75)
    nx.draw_networkx_edges(g, pos, alpha=0.25, width=0.5)
    plt.title("GRASP_Quant_Lotto_v2 — agregirani graf (svi snapshoti)")
    plt.axis("off")
    plt.tight_layout()
    path = out_dir / "aggregate_graph.png"
    plt.savefig(path, dpi=150)
    print(f"[plot] Sačuvano: {path}")
    if show_window:
        plt.show()
    plt.close()


def _detect_draw_columns(df: pd.DataFrame) -> list[str]:
    nums = [f"Num{i}" for i in range(1, 8)]
    if all(c in df.columns for c in nums):
        return nums
    base = [f"Num{i}" for i in range(1, 7)]
    if all(c in df.columns for c in base) and "Num7" in df.columns:
        return base + ["Num7"]
    return list(df.columns[:7])


def load_snapshots(csv_path: Path) -> list[nx.Graph]:
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = _detect_draw_columns(df)
    snapshots: list[nx.Graph] = []
    for _, row in df.iterrows():
        g = nx.Graph()
        g.add_nodes_from(NODES)
        nums = sorted(int(row[c]) for c in cols)
        for u, v in combinations(nums, 2):
            g.add_edge(u, v)
        snapshots.append(g)
    return snapshots


def aggregate_graph(snapshots: list[nx.Graph]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(NODES)
    for snap in snapshots:
        g.add_edges_from(snap.edges())
    return g


def all_pairs() -> list[tuple[int, int]]:
    return [(u, v) for u in NODES for v in NODES if u < v]


def labels_union_edges(pairs: list[tuple[int, int]], snapshots: list[nx.Graph]) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for snap in snapshots:
        for a, b in snap.edges():
            x, y = (a, b) if a < b else (b, a)
            edge_set.add((x, y))
    return np.array([1 if (u, v) in edge_set else 0 for u, v in pairs], dtype=np.int64)


def features_matrix(G: nx.Graph, pairs: list[tuple[int, int]]) -> np.ndarray:
    from networkx.algorithms.link_prediction import (
        adamic_adar_index,
        jaccard_coefficient,
        preferential_attachment,
    )

    cn = {(u, v): len(list(nx.common_neighbors(G, u, v))) for u, v in pairs}
    jc = {(u, v): p for u, v, p in jaccard_coefficient(G, pairs)}
    aa = {(u, v): p for u, v, p in adamic_adar_index(G, pairs)}
    pa = {(u, v): p for u, v, p in preferential_attachment(G, pairs)}
    return np.array([[cn[p], jc[p], aa[p], pa[p]] for p in pairs], dtype=np.float64)


def train_rf(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X, y)
    return rf


def run_node2vec(g: nx.Graph, seed: int) -> dict[int, np.ndarray]:
    n2v = Node2Vec(
        g,
        dimensions=N2V_DIM,
        walk_length=N2V_WALK_LEN,
        num_walks=N2V_NUM_WALKS,
        workers=1,
        seed=seed,
    )
    model = n2v.fit(window=N2V_WINDOW, min_count=1)
    emb: dict[int, np.ndarray] = {}
    for n in g.nodes():
        emb[int(n)] = np.asarray(model.wv[str(int(n))], dtype=np.float64)
    return emb


def edge_emb_mean(emb: dict[int, np.ndarray], u: int, v: int) -> float:
    return float(np.mean(np.concatenate([emb[u], emb[v]])))


def structural_edge_scores(
    rf: RandomForestRegressor,
    X: np.ndarray,
    pairs: list[tuple[int, int]],
    emb: dict[int, np.ndarray],
) -> dict[tuple[int, int], float]:
    edge_score: dict[tuple[int, int], float] = {}
    for i, (u, v) in enumerate(pairs):
        s_struct = float(rf.predict(X[i].reshape(1, -1))[0])
        s_emb = edge_emb_mean(emb, u, v)
        edge_score[(u, v)] = s_struct + s_emb
    return edge_score


def grasp_best_combo(
    edge_score: dict[tuple[int, int], float],
    top_edges: list[tuple[tuple[int, int], float]],
) -> tuple[tuple[int, ...], float]:
    candidate_nodes = sorted({n for (u, v), _ in top_edges for n in (u, v)})
    best_combo: tuple[int, ...] | None = None
    best_score = -1e18
    for combo in combinations(candidate_nodes, 7):
        scores = [
            edge_score.get((u, v), edge_score.get((v, u), 0.0))
            for u, v in combinations(combo, 2)
        ]
        score = float(np.mean(scores))
        if score > best_score or (
            np.isclose(score, best_score) and best_combo is not None and combo < best_combo
        ):
            best_score = score
            best_combo = combo
    assert best_combo is not None
    return best_combo, best_score


def top_edges_list(
    edge_score: dict[tuple[int, int], float],
    k: int,
) -> list[tuple[tuple[int, int], float]]:
    return sorted(edge_score.items(), key=lambda x: (-x[1], x[0]))[:k]


def quantum_circuit_score_deterministic(features: np.ndarray) -> float:
    n_qubits = len(features)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(features[i]), i)
        qc.rz(float(features[i]) / 2, i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    sv = Statevector.from_instruction(qc)
    ones_index = 2**n_qubits - 1
    return float(np.abs(sv.data[ones_index]) ** 2)


def quantum_scores_array(X: np.ndarray) -> np.ndarray:
    col_max = np.maximum(np.max(X, axis=0), 1e-12)
    X_norm = X / col_max * (np.pi / 2)
    return np.array([quantum_circuit_score_deterministic(f) for f in X_norm], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRASP_Quant_Lotto_v2")
    ap.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-edges", type=int, default=TOP_EDGES_DEFAULT)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Sačuvaj PNG agregiranog grafa u --out-dir",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Uz --plot: otvori prozor nakon čuvanja",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Folder za PNG (podrazumevano kurzor/GRASP_Quant_Lotto_v2_out)",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"Greška: nema CSV: {args.csv}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = all_pairs()
    snapshots = load_snapshots(args.csv)
    g = aggregate_graph(snapshots)
    maybe_plot_aggregate(g, args.plot, args.show, args.out_dir, args.seed)
    y = labels_union_edges(pairs, snapshots)
    X = features_matrix(g, pairs)
    pair_index = {p: i for i, p in enumerate(pairs)}

    rf = train_rf(X, y, args.seed)
    emb = run_node2vec(g, args.seed)
    edge_score = structural_edge_scores(rf, X, pairs, emb)

    top_edges = top_edges_list(edge_score, args.top_edges)
    best_combo, best_score = grasp_best_combo(edge_score, top_edges)

    print()
    print("PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (CSV ceo):")
    print(best_combo)
    print("Skor:", best_score)
    print()

    q_scores = quantum_scores_array(X)
    combined_score: dict[tuple[int, int], float] = {}
    for i, p in enumerate(pairs):
        combined_score[p] = ALPHA * edge_score[p] + BETA * q_scores[i]
    combined_score_array = np.array([combined_score[p] for p in pairs])

    print("Prvih 10 kombinovanih skorova (edge_score + q_score) - deterministicki:")
    for i in range(10):
        print(f"Par {pairs[i]}: {combined_score_array[i]}")
    print()

    combined_grasp_score: dict[tuple[int, int], float] = {}
    for (u, v), _ in top_edges:
        idx = pair_index[(u, v)]
        combined_grasp_score[(u, v)] = edge_score[(u, v)] + GAMMA * q_scores[idx]

    best_combo_q, best_score_q = grasp_best_combo(combined_grasp_score, top_edges)

    print("PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (SA KVANTNIM SCORE-OM - DETERMINISTICKI):")
    print(best_combo_q)
    print("Skor:", best_score_q)
    print()


if __name__ == "__main__":
    main()




"""
Computing transition probabilities: 100%|█| 39/39 [00:00<00:00
Generating walks (CPU: 1): 100%|█| 50/50 [00:00<00:00, 905.37i

PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (CSV ceo):
(4, 7, 19, 24, 26, 34, 37)
Skor: 0.9906696034174404

Prvih 10 kombinovanih skorova (edge_score + q_score) - deterministicki:
Par (1, 2): 1.0787428823368828
Par (1, 3): 1.080052336371864
Par (1, 4): 1.0827449014541344
Par (1, 5): 1.0777485528360558
Par (1, 6): 1.0791276500749518
Par (1, 7): 1.08084679654894
Par (1, 8): 1.076692641399859
Par (1, 9): 1.079335309781527
Par (1, 10): 1.0793207439692196
Par (1, 11): 1.0778930712531292

PREDIKCIJA SLEDECE LOTO7 KOMBINACIJE (SA KVANTNIM SCORE-OM - DETERMINISTICKI):
(4, 7, 19, 24, 26, 34, 37)
Skor: 1.0531696034174405
"""



"""
ispravno učitavanje kolona (Num1–Num7), 
deterministički RF (n_jobs=1), 
brzi pair_to_idx, 
zaštita X_norm, 
argparse 



Šta v2 menja:

CSV: koriste se kolone Num1–Num7 (ne row.values sa svim kolonama).
Podrazumevani CSV: GHQ/data/loto7hh_4584_k23.csv (preko Path(__file__)).
Determinizam: RandomForestRegressor(..., n_jobs=1), Node2Vec workers=1, isti --seed (39).
Brzina: pair_index umesto pairs.index u petlji za kvantni GRASP.
X_norm: np.maximum(..., 1e-12) da nema deljenja nulom.
GRASP: kod istog skora koristi se np.isclose + leksikografski combo (stabilnije od == na floatu).
Struktura: argparse (--csv, --seed, --top-edges), funkcije + main.

Pokretanje iz kurzor: python3 GRASP_Quant_Lotto_v2.py
python3 GRASP_Quant_Lotto_v2.py --plot
python3 GRASP_Quant_Lotto_v2.py --plot --show

Potrebni paketi: (qiskit, qiskit.quantum_info, node2vec, sklearn, …).
"""
