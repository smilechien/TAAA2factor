# ============================================================
# TAAAfromdigittotext_with_predpatient.py (FULLY WORKING, single-file, pasteable)
# Digital -> Tokens (Mode A/B/C) -> Co-word -> FLCA(one-link)
# -> SS(i) + Kano + Networks
# -> TAAA Round1/Round2 + Kappa + ROC + Logistic regression (WITH WALD)
# -> Heatmap: top-10 SS(i) persons per theme
# -> Scale validation + reliability + PA scree + AAC(top3 eigenvalues)
# -> (Optional) EFA rotated loadings (if factor_analyzer installed)
# -> UPDATED paired "ANOVA": within each theme (or within each TRUE_LABEL),
#    compare TestletA vs TestletB using paired t-test (F=t^2), BH-adjusted p
# -> BASELINES: ARGMAX / KMEANS / AUGMAX / KMEANS_AUGMAX
# -> NEW: Add ONE "prediction-only" dummy patient, excluded from training.
#    After BEST mode is chosen by Kappa, predict dummy patient's theme,
#    and overlay it on BEST-mode PERSON Kano plot (highlighted in red).
# ============================================================

import os
import math
import json
import time
import itertools
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# graph + stats
import networkx as nx
from scipy import stats
from pathlib import Path

def read_input_table(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)

    # CSV/TSV/TXT：用 python engine 猜分隔符（逗號/Tab 都可）
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig", sep=None, engine="python")

def init_output_dirs(out_dir: str):
    global OUT_DIR, FIG_DIR
    OUT_DIR = str(Path(out_dir).resolve())
    FIG_DIR = str(Path(OUT_DIR) / "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

# optional: Hungarian mapping
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_HUNGARIAN = True
except Exception:
    _HAVE_HUNGARIAN = False

# optional: logistic regression w/ Wald via statsmodels
try:
    import statsmodels.api as sm
    _HAVE_STATSMODELS = True
except Exception:
    _HAVE_STATSMODELS = False

# optional: EFA
try:
    from factor_analyzer import FactorAnalyzer
    _HAVE_FACTOR_ANALYZER = True
except Exception:
    _HAVE_FACTOR_ANALYZER = False

# optional: KMeans
try:
    from sklearn.cluster import KMeans
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# -------------------------
# 1) USER CONFIG
# -------------------------
np.random.seed(123)

OUT_DIR = os.path.join(os.getcwd(), "flca_taaa_html_report")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- NEW: prediction-only dummy patient ----
ENABLE_PRED_PATIENT = True
PRED_PATIENT_ID = "PRED_PATIENT"

# (Option) Provide YOUR OWN dummy patient numeric inputs:
# - Use a dict (recommended) OR a 1-row DataFrame.
# - Keys should match item columns (e.g., Math, Physics, ...).
# - Missing items will be filled by column means.
# - If None: the FIRST row of loaded dataset is used as prediction-only patient
#   (and removed from training).
PRED_PATIENT_INPUT = {
    "Math": 70, "Physics": 68, "Chemistry": 60, "Programming": 72, "Sports": 55,
    # "Literature": 80, "Linguistics": 78, "History": 75, "Arts": 82
}

# Dummy generation strength if dataset is too small to steal first row
PRED_NOISE_SD_FRAC = 0.05


GLOSSARY_LINES = [
    "Per column threshold = mean(x). x>=mean => 1 else 0; NA treated as 0.",
    "1 => Y_col; 0 => N_col",
    "Mode A: Y_ only; Mode B: N_ only; Mode C: Y_ + N_",
    "Co-word edges: within-row token pair co-occurrence counts (WCD)",
    "FLCA: one-link follower->leader + adjust to target_clusters",
    "TERM Kano edges = post-FLCA one-link relations (unique follower edge)",
    "PERSON Kano edges are optional; for interpretability we color PERSON Kano by theme (TAAA theme2_cluster).",
    "TAAA R1: mode(cluster), tie => smaller id; R2 fills missing themes",
    "support1 = max vote count of winning cluster in that row",
    "Logistic added: TRUE_LABEL ~ theme (TAAA1/TAAA2) ; theme ~ SS(i) (OvR) ; TRUE_LABEL ~ SS(i).",
    "AUGMAX baseline: per-testlet score = mean(z_items in testlet) * mean(sd(item)).",
    "KMEANS_AUGMAX baseline: kmeans on AUGMAX feature matrix (distance-based).",
    "UPDATED ANOVA: within each theme (or within each TRUE_LABEL), compare TestletA vs TestletB using paired test (F=t^2).",
    "BASELINE ARGMAX uses MEAN of standardized items within factor (not sum) to avoid factor-size bias.",
    "NEW: One dummy patient is excluded from training; predicted theme is overlaid on BEST mode only (highlighted in red).",
]


# -------------------------
# 2) Load data (FILE or DEMO)
# Expect columns: ID + numeric items + Profile(TRUE_LABEL)
# -------------------------
def load_or_demo(input_file: Optional[str] = None) -> pd.DataFrame:
    if input_file and os.path.exists(input_file):
        return read_input_table(input_file)

    demo_tsv = r"""ID	Math	Physics	Chemistry	Programming	Sports	Literature	Linguistics	History	Arts	Profile
M01	80	80	33	33	80	80	40	40	40	0
M02	80.1	80.1	33	33	80.1	40.1	40.1	40.1	40.1	0
M03	80.2	80.2	80.2	80.2	80.2	80	40.2	40.2	40.2	0
M04	80.3	80.3	80.3	80.3	33	40.3	40.3	40.3	40.3	0
M05	80.4	80.4	80.4	80.4	33	40.4	80	40.4	40.4	0
M06	80.5	80.5	80.5	80.5	33	40.5	40.5	40.5	40.5	0
M07	80.6	80.6	80.6	33	33	40.6	40.6	40.6	40.6	0
M08	80.7	33	80.7	33	80.7	40.7	40.7	40.7	40.7	0
M09	80.8	33	80.8	80.8	80.8	40.8	80	40.8	40.8	0
M10	80.9	80.9	80.9	80.9	80.9	40.9	40.9	40.9	40.9	0
M11	81	81	81	81	81	41	80	41	41	0
M12	81.1	33	81.1	33	81.1	80	41.1	41.1	41.1	0
M13	81.2	33	81.2	33	81.2	41.2	41.2	41.2	41.2	0
M14	81.3	81.3	81.3	81.3	33	80	41.3	80	41.3	0
M15	81.4	81.4	81.4	81.4	33	80	41.4	41.4	41.4	0
M16	81.5	81.5	81.5	33	81.5	41.5	80	41.5	80	0
M17	81.6	81.6	81.6	33	81.6	41.6	41.6	41.6	41.6	0
M18	33	33	81.7	81.7	81.7	80	80	41.7	80	0
M19	33	33	81.8	81.8	81.8	80	41.8	41.8	41.8	0
M20	81.9	81.9	81.9	81.9	81.9	41.9	80	41.9	41.9	0
F01	40	40	79	40	40	80	80	80	80	1
F02	40.1	40.1	79	40.1	40.1	80.1	80.1	80.1	80.1	1
F03	40.2	40.2	79	40.2	40.2	80.2	80.2	80.2	80.2	1
F04	40.3	40.3	79	40.3	40.3	80.3	40.7	80.3	40.7	1
F05	40.4	40.4	79	40.4	40.4	80.4	80.4	80.4	80.4	1
F06	40.5	40.5	79	40.5	40.5	80.5	80.5	80.5	80.5	1
F07	40.6	40.6	79	40.6	40.6	80.6	40.7	80.6	40.7	1
F08	40.7	79	40.7	40.7	40.7	80.7	80.7	80.7	80.7	1
F09	40.8	79	40.8	40.8	40.8	40.7	80.8	80.8	80.8	1
F10	40.9	79	40.9	40.9	40.9	80.9	80.9	80.9	80.9	1
F11	41	79	41	41	41	81	81	81	81	1
F12	41.1	79	41.1	41.1	41.1	81.1	40.7	81.1	81.1	1
F13	41.2	41.2	41.2	79	41.2	81.2	81.2	81.2	81.2	1
F14	41.3	41.3	41.3	79	41.3	81.3	81.3	81.3	81.3	1
F15	41.4	41.4	41.4	79	41.4	81.4	81.4	40.7	81.4	1
F16	41.5	41.5	41.5	79	41.5	81.5	81.5	81.5	81.5	1
F17	41.6	79	41.6	41.6	41.6	81.6	40.7	81.6	81.6	1
F18	41.7	79	41.7	41.7	41.7	81.7	81.7	81.7	40.7	1
F19	41.8	41.8	41.8	41.8	41.8	81.8	40.7	81.8	81.8	1
F20	41.9	41.9	41.9	41.9	41.9	81.9	81.9	81.9	81.9	1
"""
    from io import StringIO
    df = pd.read_csv(StringIO(demo_tsv), sep="\t", dtype=str)
    return df


# -------------------------
# 3) Dummy patient creation
# -------------------------
def _coerce_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_dummy_patient_from_input(df_train: pd.DataFrame, item_cols: List[str],
                                  input_obj: Any, pred_id: str, profile_col: str) -> pd.DataFrame:
    x = _coerce_numeric_df(df_train, item_cols)[item_cols]
    mu = x.mean(axis=0, skipna=True).fillna(0.0)

    vals = mu.copy()
    if isinstance(input_obj, pd.DataFrame):
        if len(input_obj) != 1:
            raise ValueError("PRED_PATIENT_INPUT DataFrame must be 1 row.")
        input_obj = input_obj.iloc[0].to_dict()

    if not isinstance(input_obj, dict):
        raise ValueError("PRED_PATIENT_INPUT must be dict, 1-row DataFrame, or None.")

    for k, v in input_obj.items():
        if k in vals.index:
            vv = pd.to_numeric(v, errors="coerce")
            if np.isfinite(vv):
                vals.loc[k] = float(vv)

    row = pd.DataFrame([vals.to_dict()])
    row.insert(0, "ID", pred_id)
    row[profile_col] = np.nan
    return row[df_train.columns]


def make_dummy_patient_row(df_train: pd.DataFrame, item_cols: List[str],
                           pred_id: str, profile_col: str, noise_sd_frac: float) -> pd.DataFrame:
    x = _coerce_numeric_df(df_train, item_cols)[item_cols]
    mu = x.mean(axis=0, skipna=True).fillna(0.0)
    sdv = x.std(axis=0, skipna=True).fillna(0.0)

    vals = mu.values.astype(float)
    if np.isfinite(noise_sd_frac) and noise_sd_frac > 0:
        vals = vals + np.random.normal(0.0, sdv.values * noise_sd_frac, size=len(vals))

    row = pd.DataFrame([dict(zip(item_cols, vals))])
    row.insert(0, "ID", pred_id)
    row[profile_col] = np.nan
    return row[df_train.columns]


# -------------------------
# 4) Tokenizer (Mode A/B/C)
# -------------------------
def make_tokens_YN(df_num: pd.DataFrame, mode: str,
                  thresholds: Optional[pd.Series] = None,
                  thr_method: str = "mean",
                  na_as_zero: bool = True) -> Tuple[Dict[str, List[str]], pd.Series]:
    assert mode in ("A", "B", "C")
    mat = df_num.values.astype(float)
    cols = df_num.columns.tolist()
    idx = df_num.index.tolist()

    if thresholds is None:
        if thr_method == "mean":
            thr = np.nanmean(mat, axis=0)
        else:
            thr = np.nanmedian(mat, axis=0)
        thresholds = pd.Series(thr, index=cols)
    else:
        thresholds = thresholds.reindex(cols).astype(float)

    tokens_list: Dict[str, List[str]] = {}
    for i, rid in enumerate(idx):
        out = []
        for j, col in enumerate(cols):
            v = mat[i, j]
            t = thresholds[col]
            if not np.isfinite(t):
                continue
            if not np.isfinite(v):
                if na_as_zero:
                    tok = f"N_{col}"
                    if mode in ("B", "C"):
                        out.append(tok)
                continue
            if v >= t:
                tok = f"Y_{col}"
                if mode in ("A", "C"):
                    out.append(tok)
            else:
                tok = f"N_{col}"
                if mode in ("B", "C"):
                    out.append(tok)
        tokens_list[rid] = sorted(set(out))
    return tokens_list, thresholds


def tokenize_one_row_YN(one_row: pd.Series, thresholds: pd.Series, mode: str,
                        na_as_zero: bool = True) -> List[str]:
    df1 = pd.DataFrame([one_row]).reindex(columns=thresholds.index)
    for c in df1.columns:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")
    tokens, _ = make_tokens_YN(df1, mode=mode, thresholds=thresholds, na_as_zero=na_as_zero)
    return tokens[df1.index[0]]


# -------------------------
# 5) Co-word build (nodes + edges)
# -------------------------
def build_coword(tokens_list: Dict[str, List[str]], min_wcd: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_counts: Dict[str, int] = {}
    edge_counts: Dict[Tuple[str, str], int] = {}

    for _, toks in tokens_list.items():
        if not toks:
            continue
        toks = sorted(set(toks))
        for u in toks:
            node_counts[u] = node_counts.get(u, 0) + 1
        if len(toks) >= 2:
            for a, b in itertools.combinations(toks, 2):
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    nodes = pd.DataFrame({"name": list(node_counts.keys()),
                          "value": list(node_counts.values())})
    nodes = nodes.sort_values(["value", "name"], ascending=[False, True]).reset_index(drop=True)

    if not edge_counts:
        edges = pd.DataFrame({"term1": [], "term2": [], "wcd": []})
        return nodes, edges

    edges = pd.DataFrame([(a, b, w) for (a, b), w in edge_counts.items()],
                         columns=["term1", "term2", "wcd"])
    edges = edges[edges["wcd"] >= int(min_wcd)].copy()
    edges = edges.sort_values(["wcd", "term1", "term2"], ascending=[False, True, True]).reset_index(drop=True)
    return nodes, edges


# -------------------------
# 6) FLCA + adjust to target_clusters + ONE-LINK relations
# -------------------------
def flca_cluster_with_onelink(nodes: pd.DataFrame, edges: pd.DataFrame,
                             target_clusters: int, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = nodes.copy()
    nodes["name"] = nodes["name"].astype(str)
    node_value = dict(zip(nodes["name"], pd.to_numeric(nodes["value"], errors="coerce").fillna(0).astype(float)))

    # build neighbor weights from edges
    neigh_w: Dict[str, Dict[str, float]] = {n: {} for n in nodes["name"]}
    for _, r in edges.iterrows():
        a = str(r["term1"]); b = str(r["term2"]); w = float(r["wcd"])
        neigh_w.setdefault(a, {})[b] = neigh_w.get(a, {}).get(b, 0.0) + w
        neigh_w.setdefault(b, {})[a] = neigh_w.get(b, {}).get(a, 0.0) + w

    def choose_leader(v: str) -> Tuple[str, float]:
        nb = neigh_w.get(v, {})
        if not nb:
            return v, 0.0
        mx = max(nb.values())
        cand = [k for k, ww in nb.items() if ww == mx]
        if len(cand) > 1:
            vv = [node_value.get(c, 0.0) for c in cand]
            m2 = max(vv)
            cand = [c for c in cand if node_value.get(c, 0.0) == m2]
        leader = sorted(cand)[0]
        return leader, float(mx)

    leader_of = {}
    wcd_of = {}
    for v in nodes["name"]:
        leader, wcd = choose_leader(v)
        leader_of[v] = leader
        wcd_of[v] = wcd

    def find_root(v: str) -> str:
        seen = []
        cur = v
        for _ in range(2000):
            if cur in seen:
                cyc = seen[seen.index(cur):]
                return sorted(cyc)[0]
            seen.append(cur)
            nxt = leader_of.get(cur, cur)
            if nxt == cur:
                return cur
            cur = nxt
        return v

    nodes["Leader"] = nodes["name"].map(leader_of)
    nodes["RootLeader"] = nodes["name"].map(find_root)

    roots = nodes["RootLeader"].unique().tolist()
    root_val = {r: float(nodes.loc[nodes["RootLeader"] == r, "value"].astype(float).sum()) for r in roots}
    roots_sorted = sorted(roots, key=lambda r: (-root_val.get(r, 0.0), r))
    root2cid = {r: i + 1 for i, r in enumerate(roots_sorted)}
    nodes["cluster"] = nodes["RootLeader"].map(root2cid).astype(int)

    def relabel_by_total(nod: pd.DataFrame) -> pd.DataFrame:
        cc = sorted(nod["cluster"].unique().tolist())
        tot = {c: float(nod.loc[nod["cluster"] == c, "value"].astype(float).sum()) for c in cc}
        cc2 = sorted(cc, key=lambda c: (-tot.get(c, 0.0), c))
        mp = {old: i + 1 for i, old in enumerate(cc2)}
        nod = nod.copy()
        nod["cluster"] = nod["cluster"].map(mp).astype(int)
        return nod

    def inter_w(nod: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        if edges.empty:
            return {}
        cmap = dict(zip(nod["name"], nod["cluster"]))
        out: Dict[Tuple[int, int], float] = {}
        for _, r in edges.iterrows():
            c1 = cmap.get(str(r["term1"]))
            c2 = cmap.get(str(r["term2"]))
            if c1 is None or c2 is None or c1 == c2:
                continue
            a, b = (c1, c2) if c1 < c2 else (c2, c1)
            out[(a, b)] = out.get((a, b), 0.0) + float(r["wcd"])
        return out

    nodes = relabel_by_total(nodes)
    if verbose:
        print(f"[FLCA] initial clusters: {nodes['cluster'].nunique()}")

    # merge if too many clusters
    while nodes["cluster"].nunique() > int(target_clusters):
        nodes = relabel_by_total(nodes)
        cc = sorted(nodes["cluster"].unique().tolist())
        tot = {c: float(nodes.loc[nodes["cluster"] == c, "value"].astype(float).sum()) for c in cc}
        smallest = min(cc, key=lambda c: tot.get(c, 0.0))
        others = [c for c in cc if c != smallest]
        if not others:
            break
        iw = inter_w(nodes)
        best = others[0]
        bestw = -1e18
        for j in others:
            a, b = (smallest, j) if smallest < j else (j, smallest)
            wsum = iw.get((a, b), 0.0)
            if wsum > bestw:
                bestw = wsum
                best = j
        nodes.loc[nodes["cluster"] == smallest, "cluster"] = best

    # split if too few clusters
    while nodes["cluster"].nunique() < int(target_clusters):
        nodes = relabel_by_total(nodes)
        cc = sorted(nodes["cluster"].unique().tolist())
        tot = {c: float(nodes.loc[nodes["cluster"] == c, "value"].astype(float).sum()) for c in cc}
        largest = max(cc, key=lambda c: tot.get(c, 0.0))
        members = nodes.loc[nodes["cluster"] == largest].copy()
        members = members.sort_values(["value", "name"], ascending=[False, True])
        if len(members) < 2:
            break
        half = int(math.ceil(len(members) / 2))
        new_id = int(nodes["cluster"].max()) + 1
        move_names = members["name"].iloc[:half].tolist()
        nodes.loc[nodes["name"].isin(move_names), "cluster"] = new_id

    nodes = relabel_by_total(nodes)
    if verbose:
        print(f"[FLCA] final clusters: {nodes['cluster'].nunique()}")

    rel = pd.DataFrame({
        "follower": nodes["name"].astype(str),
        "leader": nodes["name"].astype(str).map(leader_of),
        "wcd": nodes["name"].astype(str).map(lambda x: int(round(wcd_of.get(x, 0.0))))
    })
    rel = rel[rel["follower"] != rel["leader"]].copy()
    rel = rel.sort_values(["wcd", "follower", "leader"], ascending=[False, True, True]).reset_index(drop=True)
    return nodes, rel


# -------------------------
# 7) Silhouette-like SS(i) on graph distances + a*
# -------------------------
def compute_silhouette_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    if len(nodes) < 3 or edges is None or edges.empty:
        nodes["a_i"] = 0.0
        nodes["b_i"] = 0.0
        nodes["SS_i"] = 0.0
        nodes["a_star"] = 1.0
        return nodes

    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["name"]))
    for _, r in edges.iterrows():
        a = str(r["term1"]); b = str(r["term2"])
        wcd = float(r["wcd"])
        distw = 1.0 / (1.0 + wcd)
        if G.has_edge(a, b):
            # keep smaller distance if duplicated
            G[a][b]["weight"] = min(G[a][b]["weight"], distw)
        else:
            G.add_edge(a, b, weight=distw)

    # all-pairs shortest paths
    dists = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    names = nodes["name"].astype(str).tolist()
    n = len(names)
    D = np.full((n, n), np.nan, dtype=float)

    for i, u in enumerate(names):
        du = dists.get(u, {})
        for j, v in enumerate(names):
            if u == v:
                D[i, j] = 0.0
            else:
                D[i, j] = du.get(v, np.nan)

    fin = D[np.isfinite(D)]
    mx = float(fin.max()) if fin.size else 1.0
    D[~np.isfinite(D)] = mx * 1.5

    cl = pd.to_numeric(nodes["cluster"], errors="coerce").fillna(0).astype(int).values
    a_i = np.zeros(n, dtype=float)
    b_i = np.zeros(n, dtype=float)
    for i in range(n):
        same = np.where((cl == cl[i]) & (np.arange(n) != i))[0]
        a_i[i] = float(D[i, same].mean()) if same.size else 0.0

        other_cls = sorted(set(cl.tolist()) - {cl[i]})
        if not other_cls:
            b_i[i] = 0.0
        else:
            bvals = []
            for cc in other_cls:
                idx = np.where(cl == cc)[0]
                bvals.append(float(D[i, idx].mean()) if idx.size else np.inf)
            b_i[i] = float(min(bvals))

    den = np.maximum(a_i, b_i)
    SS_i = np.where(den == 0, 0.0, (b_i - a_i) / den)
    a_star = 1.0 / (1.0 + a_i)

    nodes["a_i"] = a_i
    nodes["b_i"] = b_i
    nodes["SS_i"] = SS_i
    nodes["a_star"] = a_star
    return nodes


# -------------------------
# 8) Majority sampling Top-k (quota per cluster)
# -------------------------
def majority_topk(nodes: pd.DataFrame, k: int = 20) -> List[str]:
    cc = sorted(nodes["cluster"].unique().tolist())
    tot = np.array([nodes.loc[nodes["cluster"] == c, "value"].astype(float).sum() for c in cc], dtype=float)
    tot_sum = tot.sum() if np.isfinite(tot).any() else 1.0
    frac = tot / (tot_sum if tot_sum > 0 else 1.0)

    alloc = np.maximum(1, np.floor(k * frac).astype(int))
    while alloc.sum() < k:
        j = int(np.argmax(frac - alloc / max(1, k)))
        alloc[j] += 1
    while alloc.sum() > k:
        j = int(np.argmax(alloc))
        if alloc[j] > 1:
            alloc[j] -= 1
        else:
            break

    picked = []
    for i, c in enumerate(cc):
        sub = nodes.loc[nodes["cluster"] == c].copy()
        sub = sub.sort_values(["value", "name"], ascending=[False, True])
        take = min(int(alloc[i]), len(sub))
        picked.extend(sub["name"].astype(str).iloc[:take].tolist())

    picked = list(dict.fromkeys(picked))
    if len(picked) > k:
        sub = nodes.loc[nodes["name"].isin(picked)].copy()
        sub = sub.sort_values(["value", "name"], ascending=[False, True])
        picked = sub["name"].astype(str).iloc[:k].tolist()
    return picked


# -------------------------
# 9) TAAA Round1 + Round2
# -------------------------
def taaa_rounds(tokens_list: Dict[str, List[str]],
                node_cluster_map: Dict[str, int],
                K: int,
                cluster_leader_map: Dict[int, str]) -> pd.DataFrame:
    ids = list(tokens_list.keys())
    theme1_cluster = np.zeros(len(ids), dtype=int)
    support1 = np.zeros(len(ids), dtype=int)
    counts_list: List[Dict[int, int]] = []

    for i, pid in enumerate(ids):
        tks = [t for t in tokens_list[pid] if t in node_cluster_map]
        if not tks:
            counts_list.append({})
            continue
        cc = [int(node_cluster_map[t]) for t in tks]
        tb = {}
        for c in cc:
            tb[c] = tb.get(c, 0) + 1
        mx = max(tb.values())
        cand = [c for c, v in tb.items() if v == mx]
        theme1_cluster[i] = min(cand)
        support1[i] = int(mx)
        counts_list.append(tb)

    theme2_cluster = theme1_cluster.copy()

    present = sorted(set(theme2_cluster[theme2_cluster > 0].tolist()))
    missing = [m for m in range(1, K + 1) if m not in present]
    if missing:
        for m in missing:
            cand_idx = []
            for i in range(len(ids)):
                tb = counts_list[i]
                if not tb:
                    continue
                if m in tb and theme2_cluster[i] != m:
                    cand_idx.append(i)
            if not cand_idx:
                continue
            score = []
            for i in cand_idx:
                tb = counts_list[i]
                count_m = tb.get(m, 0)
                th1 = theme1_cluster[i]
                count_theme1 = tb.get(th1, 0) if th1 > 0 else 0
                delta = count_m - count_theme1
                score.append((delta, count_m, support1[i], i))
            score.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
            theme2_cluster[score[0][3]] = m

    def th_name(c: int) -> str:
        if c <= 0:
            return "Unassigned"
        return cluster_leader_map.get(c, f"Theme_{c}")

    theme1 = [th_name(int(c)) for c in theme1_cluster]
    theme2 = [th_name(int(c)) for c in theme2_cluster]

    df = pd.DataFrame({
        "ID": ids,
        "theme1_cluster": theme1_cluster,
        "theme1": theme1,
        "support1": support1,
        "theme2_cluster": theme2_cluster,
        "theme2": theme2
    })
    return df


# -------------------------
# 10) Kappa + mapping
# -------------------------
def cohen_kappa(tab: np.ndarray) -> float:
    N = tab.sum()
    if N <= 0:
        return float("nan")
    po = np.trace(tab) / N
    rs = tab.sum(axis=1)
    cs = tab.sum(axis=0)
    pe = float((rs * cs).sum()) / (N ** 2)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def best_mapping(tab: np.ndarray) -> List[int]:
    # returns perm p where columns are permuted tab[:, p] to maximize diagonal sum
    k = tab.shape[0]
    if _HAVE_HUNGARIAN:
        mx = tab.max() if tab.size else 0
        cost = mx - tab
        r, c = linear_sum_assignment(cost)
        # columns assignment in row order
        perm = list(c.tolist())
        return perm
    # brute force fallback (k<=6 recommended)
    if k > 6:
        raise RuntimeError("Install scipy for Hungarian mapping (k>6 too large for brute force).")
    bestp = list(range(k))
    bestv = -1
    for p in itertools.permutations(range(k)):
        v = sum(tab[i, p[i]] for i in range(k))
        if v > bestv:
            bestv = v
            bestp = list(p)
    return bestp


def eval_kxk(true_label: np.ndarray, pred_cluster: np.ndarray, K: int) -> Dict[str, Any]:
    y = pd.to_numeric(pd.Series(true_label), errors="coerce")
    p = pd.to_numeric(pd.Series(pred_cluster), errors="coerce")
    ok = y.notna() & p.notna() & (p.astype(int) > 0)
    y = y[ok].astype(int).values
    p = p[ok].astype(int).values

    # map true labels to 1..K by sorted unique
    true_levels = sorted(set(y.tolist()))
    if len(true_levels) < K:
        # pad if needed (rare)
        mx = max(true_levels) if true_levels else 0
        true_levels += list(range(mx + 1, mx + 1 + (K - len(true_levels))))
    true_levels = true_levels[:K]
    y2 = np.array([true_levels.index(v) + 1 if v in true_levels else 1 for v in y], dtype=int)

    tab = np.zeros((K, K), dtype=int)
    for yy, pp in zip(y2, p):
        if 1 <= yy <= K and 1 <= pp <= K:
            tab[yy - 1, pp - 1] += 1

    perm = best_mapping(tab)
    tab_mapped = tab[:, perm]
    kap = cohen_kappa(tab_mapped)

    return {
        "tab": tab,
        "perm": perm,
        "tab_mapped": tab_mapped,
        "kappa": kap,
        "n_used": int(ok.sum())
    }


# -------------------------
# 11) ROC (score vs y01) + cutpoint by Youden J
# -------------------------
def roc_compute(score: np.ndarray, y01: np.ndarray, n_boot: int = 200, seed: int = 123) -> Dict[str, Any]:
    score = pd.to_numeric(pd.Series(score), errors="coerce").values.astype(float)
    y01 = pd.to_numeric(pd.Series(y01), errors="coerce").fillna(-1).astype(int).values
    ok = np.isfinite(score) & np.isin(y01, [0, 1])
    score = score[ok]
    y01 = y01[ok]
    if len(score) < 5 or len(set(y01.tolist())) < 2:
        return {"df": None, "auc": float("nan"), "ci": (float("nan"), float("nan")), "best": None}

    thr = np.unique(score)[::-1]
    thr = np.concatenate(([np.inf], thr, [-np.inf]))

    rows = []
    for t in thr:
        pred = (score >= t).astype(int)
        TP = int(((pred == 1) & (y01 == 1)).sum())
        FP = int(((pred == 1) & (y01 == 0)).sum())
        FN = int(((pred == 0) & (y01 == 1)).sum())
        TN = int(((pred == 0) & (y01 == 0)).sum())
        sens = TP / (TP + FN) if (TP + FN) else np.nan
        spec = TN / (TN + FP) if (TN + FP) else np.nan
        fpr = FP / (FP + TN) if (FP + TN) else np.nan
        tpr = sens
        rows.append((fpr, tpr, t, sens, spec))

    df = pd.DataFrame(rows, columns=["fpr", "tpr", "thr", "sens", "spec"]).dropna()
    df = df.sort_values(["fpr", "tpr"]).reset_index(drop=True)
    if len(df) < 2:
        return {"df": df, "auc": float("nan"), "ci": (float("nan"), float("nan")), "best": None}

    # trapezoid auc
    auc = float(np.sum(np.diff(df["fpr"].values) * (df["tpr"].values[:-1] + df["tpr"].values[1:]) / 2.0))

    df2 = df.copy()
    df2["J"] = df2["sens"] + df2["spec"] - 1.0
    best = df2.iloc[int(df2["J"].values.argmax())].to_dict()

    # bootstrap CI
    ci = (float("nan"), float("nan"))
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        auc_b = []
        n = len(score)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rb = roc_compute(score[idx], y01[idx], n_boot=0)
            if np.isfinite(rb["auc"]):
                auc_b.append(rb["auc"])
        if len(auc_b) >= 20:
            ci = (float(np.quantile(auc_b, 0.025)), float(np.quantile(auc_b, 0.975)))

    return {"df": df, "auc": auc, "ci": ci, "best": best}


def plot_roc_combined(roc_internal: Dict[str, Any], roc_true: Dict[str, Any], title: str, out_path: str) -> Optional[str]:
    fig = plt.figure(figsize=(7.8, 5.4), dpi=160)
    ax = fig.add_subplot(111)

    did = False
    if roc_internal.get("df") is not None and isinstance(roc_internal["df"], pd.DataFrame) and len(roc_internal["df"]) >= 2:
        d = roc_internal["df"]
        ax.plot(d["fpr"], d["tpr"], label="Internal ROC (circular)")
        did = True
    if roc_true.get("df") is not None and isinstance(roc_true["df"], pd.DataFrame) and len(roc_true["df"]) >= 2:
        d = roc_true["df"]
        ax.plot(d["fpr"], d["tpr"], label="TRUE_LABEL")
        did = True

    ax.plot([0, 1], [0, 1], linestyle="dotted")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if did:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return os.path.basename(out_path) if did else None


# -------------------------
# 12) Logistic regression helpers (WITH WALD)
# -------------------------
def logistic_ssi_table(score: np.ndarray, y01: np.ndarray, model_name: str) -> pd.DataFrame:
    n = len(score)
    out_cols = ["Model", "Term", "N", "Beta", "SE", "z", "Wald", "OR", "CI_low", "CI_high", "p", "p_LRT"]
    if not _HAVE_STATSMODELS:
        return pd.DataFrame([[model_name, "SS(i)", n] + [np.nan]*9], columns=out_cols)

    s = pd.to_numeric(pd.Series(score), errors="coerce")
    y = pd.to_numeric(pd.Series(y01), errors="coerce").fillna(-1).astype(int)
    ok = s.notna() & y.isin([0, 1])
    s = s[ok].astype(float).values
    y = y[ok].astype(int).values

    if len(s) < 8 or len(set(y.tolist())) < 2:
        return pd.DataFrame([[model_name, "SS(i)", len(s)] + [np.nan]*9], columns=out_cols)

    X = sm.add_constant(s)
    try:
        fit = sm.Logit(y, X).fit(disp=False)
    except Exception:
        return pd.DataFrame([[model_name, "SS(i)", len(s)] + [np.nan]*9], columns=out_cols)

    beta = float(fit.params[1])
    se = float(fit.bse[1]) if np.isfinite(fit.bse[1]) else np.nan
    z = beta / se if np.isfinite(se) and se > 0 else np.nan
    wald = z*z if np.isfinite(z) else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

    zcrit = stats.norm.ppf(0.975)
    ci_low = beta - zcrit*se if np.isfinite(se) else np.nan
    ci_high = beta + zcrit*se if np.isfinite(se) else np.nan

    # LRT p-value vs null
    p_lrt = np.nan
    try:
        fit0 = sm.Logit(y, sm.add_constant(np.zeros_like(s))).fit(disp=False)
        LR = 2*(fit.llf - fit0.llf)
        p_lrt = float(1 - stats.chi2.cdf(LR, df=1))
    except Exception:
        p_lrt = np.nan

    row = [model_name, "SS(i)", int(len(s)), beta, se, z, wald, math.exp(beta),
           math.exp(ci_low) if np.isfinite(ci_low) else np.nan,
           math.exp(ci_high) if np.isfinite(ci_high) else np.nan,
           p, p_lrt]
    return pd.DataFrame([row], columns=out_cols)


def logistic_ssi_to_theme_ovr_table(score: np.ndarray, cluster: np.ndarray, prefix: str) -> pd.DataFrame:
    cluster = pd.to_numeric(pd.Series(cluster), errors="coerce").fillna(0).astype(int).values
    uniq = sorted(set(cluster.tolist()))
    uniq = [c for c in uniq if c > 0]
    if len(uniq) < 2:
        return pd.DataFrame([[prefix, "(OvR)", int(len(score))] + [np.nan]*9],
                            columns=["Model", "Term", "N", "Beta", "SE", "z", "Wald", "OR", "CI_low", "CI_high", "p", "p_LRT"])
    rows = []
    for cc in uniq:
        y = (cluster == cc).astype(int)
        if y.sum() < 3 or (y == 0).sum() < 3:
            rows.append([f"{prefix} (cluster={cc})", "SS(i)", int(len(y))] + [np.nan]*9)
        else:
            df1 = logistic_ssi_table(score, y, f"{prefix} (cluster={cc})")
            rows.extend(df1.values.tolist())
    return pd.DataFrame(rows, columns=["Model", "Term", "N", "Beta", "SE", "z", "Wald", "OR", "CI_low", "CI_high", "p", "p_LRT"])


# -------------------------
# 13) Kano plot helpers (matplotlib)
# -------------------------
def kano_layers(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    mean_x = float(np.nanmean(x))
    mean_y = float(np.nanmean(y))
    min_x, max_x = float(np.nanmin(x)), float(np.nanmax(x))
    min_y, max_y = float(np.nanmin(y)), float(np.nanmax(y))

    spread_x = (max_x - min_x) * 0.43 if max_x > min_x else 1.0
    spread_y = (max_y - min_y) * 1.2 if max_y > min_y else 1.0
    kano_amp = 1.1

    t = np.linspace(0, 1, 300)
    lower_x = t*spread_x - spread_x/2 + mean_x
    lower_y = mean_y - kano_amp*spread_y*(1 - t)**2
    upper_x = -t*spread_x + spread_x/2 + mean_x
    upper_y = mean_y + kano_amp*spread_y*(1 - t)**2

    A = kano_amp * spread_y
    r_y = 0.25 * A * 0.5
    theta = np.linspace(0, 2*np.pi, 300)
    x_span = max_x - min_x
    y_span = max_y - min_y
    r_x = min(0.45*x_span, r_y*(x_span/y_span)) if (x_span > 0 and y_span > 0) else r_y
    circle_x = mean_x + r_x*np.cos(theta)
    circle_y = mean_y + r_y*np.sin(theta)

    return {
        "lower": (lower_x, lower_y),
        "upper": (upper_x, upper_y),
        "circle": (circle_x, circle_y),
        "mean_x": mean_x,
        "mean_y": mean_y,
    }


def save_kano_term(nodes: pd.DataFrame, rel_one: pd.DataFrame, title: str, out_png: str) -> str:
    df = nodes.copy()
    df = df[np.isfinite(df["a_star"]) & np.isfinite(df["SS_i"])].copy()
    x = df["a_star"].values.astype(float)
    y = df["SS_i"].values.astype(float)
    kl = kano_layers(x, y)

    fig = plt.figure(figsize=(9.8, 5.8), dpi=160)
    ax = fig.add_subplot(111)

    ax.plot(*kl["lower"], linewidth=1.0)
    ax.plot(*kl["upper"], linewidth=1.0)
    ax.plot(*kl["circle"], linewidth=1.0)
    ax.axvline(kl["mean_x"], linestyle="dotted", linewidth=0.9)
    ax.axhline(kl["mean_y"], linestyle="dotted", linewidth=0.9)

    # edges
    if rel_one is not None and not rel_one.empty:
        m = {n: i for i, n in enumerate(df["name"].astype(str).tolist())}
        for _, r in rel_one.iterrows():
            f = str(r["follower"]); l = str(r["leader"])
            if f in m and l in m:
                i = m[f]; j = m[l]
                ax.plot([x[i], x[j]], [y[i], y[j]], alpha=0.35, linewidth=max(0.6, float(r.get("wcd", 1))/4.0))

    # points
    sizes = np.clip(df["value"].astype(float).values, 1, None)
    sizes = 25 + 3.0*np.sqrt(sizes)
    ax.scatter(x, y, s=sizes, alpha=0.85)

    # labels
    for i, nm in enumerate(df["name"].astype(str).tolist()):
        ax.text(x[i], y[i], nm, fontsize=8, fontweight="bold", ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("a* = 1/(1+a(i))")
    ax.set_ylabel("SS(i)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return os.path.basename(out_png)


def save_kano_person(pnodes: pd.DataFrame, title: str, color_col: str, out_png: str,
                     highlight_name: Optional[str] = None) -> str:
    df = pnodes.copy()
    df = df[np.isfinite(df["a_star"]) & np.isfinite(df["SS_i"])].copy()
    x = df["a_star"].values.astype(float)
    y = df["SS_i"].values.astype(float)
    kl = kano_layers(x, y)

    fig = plt.figure(figsize=(9.8, 5.8), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(*kl["lower"], linewidth=1.0)
    ax.plot(*kl["upper"], linewidth=1.0)
    ax.plot(*kl["circle"], linewidth=1.0)
    ax.axvline(kl["mean_x"], linestyle="dotted", linewidth=0.9)
    ax.axhline(kl["mean_y"], linestyle="dotted", linewidth=0.9)

    # color groups
    if color_col not in df.columns:
        df[color_col] = df["cluster"]

    groups = pd.Categorical(df[color_col].astype(str))
    codes = groups.codes

    sizes = np.clip(pd.to_numeric(df["value"], errors="coerce").fillna(1).values.astype(float), 1, None)
    sizes = 25 + 3.0*np.sqrt(sizes)
    ax.scatter(x, y, s=sizes, c=codes, alpha=0.88)

    for i, nm in enumerate(df["name"].astype(str).tolist()):
        ax.text(x[i], y[i], nm, fontsize=8, fontweight="bold", ha="center", va="bottom")

    # highlight dummy
    if highlight_name:
        hit = df["name"].astype(str) == str(highlight_name)
        if hit.any():
            i = int(np.where(hit.values)[0][0])
            ax.scatter([x[i]], [y[i]], s=260, c="red", marker="^", alpha=0.95)
            ax.text(x[i], y[i], str(highlight_name), fontsize=9, fontweight="bold",
                    color="red", ha="center", va="top")

    ax.set_title(title)
    ax.set_xlabel("a* = 1/(1+a(i))")
    ax.set_ylabel("SS(i)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return os.path.basename(out_png)


# -------------------------
# 14) PERSON graph from tokens (shared tokens edges)
# -------------------------
def build_person_graph_from_tokens(tokens_list: Dict[str, List[str]],
                                  min_shared: int = 2, top_m: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(tokens_list.keys())
    inv: Dict[str, List[str]] = {}
    for pid, tks in tokens_list.items():
        tks = sorted(set(tks))
        for tk in tks:
            inv.setdefault(tk, []).append(pid)

    edge_counts: Dict[Tuple[str, str], int] = {}
    for tk, ppl in inv.items():
        ppl = sorted(set(ppl))
        if len(ppl) < 2:
            continue
        for a, b in itertools.combinations(ppl, 2):
            if a > b:
                a, b = b, a
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    nodes = pd.DataFrame({
        "name": ids,
        "value": [len(set(tokens_list[i])) for i in ids]
    })

    if not edge_counts:
        edges = pd.DataFrame({"term1": [], "term2": [], "wcd": []})
        return nodes, edges

    edges = pd.DataFrame([(a, b, w) for (a, b), w in edge_counts.items()],
                         columns=["term1", "term2", "wcd"])
    edges = edges[edges["wcd"] >= int(min_shared)].copy()
    if edges.empty:
        return nodes, edges

    # keep top_m per node
    keep = set()
    for pid in ids:
        sub = edges[(edges["term1"] == pid) | (edges["term2"] == pid)].copy()
        sub = sub.sort_values("wcd", ascending=False).head(int(top_m))
        for _, r in sub.iterrows():
            a = str(r["term1"]); b = str(r["term2"])
            if a > b:
                a, b = b, a
            keep.add((a, b))

    mask = []
    for _, r in edges.iterrows():
        a = str(r["term1"]); b = str(r["term2"])
        if a > b:
            a, b = b, a
        mask.append((a, b) in keep)

    edges = edges[np.array(mask, dtype=bool)].copy()
    edges = edges.sort_values(["wcd", "term1", "term2"], ascending=[False, True, True]).reset_index(drop=True)
    return nodes, edges


# -------------------------
# 15) Testlets from TERM token clusters + scores
# -------------------------
def build_item_testlets_from_term_nodes(term_nodes: pd.DataFrame) -> Dict[str, Any]:
    nd = term_nodes.copy()
    nd["name"] = nd["name"].astype(str)
    nd["cluster"] = pd.to_numeric(nd["cluster"], errors="coerce").fillna(0).astype(int)
    if "value" not in nd.columns:
        nd["value"] = 1

    nd2 = nd[nd["name"].str.match(r"^[YN]_")].copy()
    if nd2.empty:
        return {"item_map": pd.DataFrame({"item": [], "cluster": []}),
                "testlet_items": {}, "testlet_labels": []}

    nd2["item"] = nd2["name"].str.replace(r"^[YN]_", "", regex=True)
    agg = nd2.groupby(["item", "cluster"], as_index=False)["value"].sum()
    agg = agg[(agg["cluster"] > 0)].copy()
    if agg.empty:
        return {"item_map": pd.DataFrame({"item": [], "cluster": []}),
                "testlet_items": {}, "testlet_labels": []}

    item_cluster = []
    for it, sub in agg.groupby("item"):
        mx = sub["value"].max()
        cand = sub[sub["value"] == mx].sort_values("cluster")
        item_cluster.append((it, int(cand["cluster"].iloc[0])))

    item_map = pd.DataFrame(item_cluster, columns=["item", "cluster"]).sort_values(["cluster", "item"])
    testlet_items = {int(cc): g["item"].tolist() for cc, g in item_map.groupby("cluster")}
    testlet_labels = [f"Testlet{cc}" for cc in sorted(testlet_items.keys())]
    return {"item_map": item_map, "testlet_items": testlet_items, "testlet_labels": testlet_labels}


def compute_testlet_scores(df_num: pd.DataFrame, testlet_obj: Dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame({"ID": df_num.index.astype(str).tolist()})
    testlet_items = testlet_obj.get("testlet_items", {})
    if not testlet_items:
        return out
    for cc in sorted(testlet_items.keys()):
        items = [it for it in testlet_items[cc] if it in df_num.columns]
        coln = f"Testlet{cc}"
        if not items:
            out[coln] = np.nan
        else:
            out[coln] = df_num[items].mean(axis=1, skipna=True).values
    return out


def compute_testlet_scores_augmax(df_num: pd.DataFrame, testlet_obj: Dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame({"ID": df_num.index.astype(str).tolist()})
    testlet_items = testlet_obj.get("testlet_items", {})
    if not testlet_items:
        return out

    df_imp = df_num.copy()
    for c in df_imp.columns:
        x = pd.to_numeric(df_imp[c], errors="coerce")
        m = float(x.mean(skipna=True)) if np.isfinite(x.mean(skipna=True)) else 0.0
        x = x.fillna(m)
        df_imp[c] = x

    X = df_imp.values.astype(float)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)

    for cc in sorted(testlet_items.keys()):
        items = [it for it in testlet_items[cc] if it in df_imp.columns]
        coln = f"Testlet{cc}"
        if not items:
            out[coln] = np.nan
            continue
        idx = [df_imp.columns.get_loc(it) for it in items]
        z_mean = Z[:, idx].mean(axis=1)
        item_sds = X[:, idx].std(axis=0, ddof=1)
        w = float(np.nanmean(item_sds)) if np.isfinite(np.nanmean(item_sds)) else 1.0
        out[coln] = z_mean * w
    return out


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(pv)
    tmp[order] = adj
    out[ok] = tmp
    return out


def build_anova_table_testlets(testlet_scores_df: pd.DataFrame, group_col: str,
                               testlet_obj: Optional[Dict[str, Any]] = None,
                               min_n: int = 3) -> pd.DataFrame:
    df = testlet_scores_df.copy()
    if group_col not in df.columns:
        return pd.DataFrame(columns=["Testlet", "Items", "N", "k", "F", "p", "eta2", "p_adj"])

    testlet_cols = sorted([c for c in df.columns if c.startswith("Testlet") and c[7:].isdigit()])
    if len(testlet_cols) < 2:
        return pd.DataFrame(columns=["Testlet", "Items", "N", "k", "F", "p", "eta2", "p_adj"])

    A, B = testlet_cols[0], testlet_cols[1]
    items_txt = ""
    if testlet_obj and testlet_obj.get("testlet_items"):
        ti = testlet_obj["testlet_items"]
        def short_items(items):
            if not items:
                return ""
            show = items[:10]
            return f"{len(items)} items: " + ", ".join(show) + (", ..." if len(items) > 10 else "")
        a_items = ti.get(int(A.replace("Testlet", "")), [])
        b_items = ti.get(int(B.replace("Testlet", "")), [])
        items_txt = f"{A} {{{short_items(a_items)}}} vs {B} {{{short_items(b_items)}}}"

    rows = []
    for g in sorted(df[group_col].dropna().astype(str).unique().tolist()):
        dg = df[df[group_col].astype(str) == g].copy()
        dg[A] = pd.to_numeric(dg[A], errors="coerce")
        dg[B] = pd.to_numeric(dg[B], errors="coerce")
        dg = dg[np.isfinite(dg[A]) & np.isfinite(dg[B])]
        n = len(dg)
        if n < min_n:
            rows.append([f"{A} vs {B} @{group_col}={g}", items_txt, int(n), 2, np.nan, np.nan, np.nan])
            continue
        tt = stats.ttest_rel(dg[A].values, dg[B].values, nan_policy="omit")
        tval = float(tt.statistic) if np.isfinite(tt.statistic) else np.nan
        dfree = n - 1
        Fval = tval*tval if np.isfinite(tval) else np.nan
        eta2 = (Fval / (Fval + dfree)) if np.isfinite(Fval) else np.nan
        rows.append([f"{A} vs {B} @{group_col}={g}", items_txt, int(n), 2,
                     Fval, float(tt.pvalue), eta2])

    out = pd.DataFrame(rows, columns=["Testlet", "Items", "N", "k", "F", "p", "eta2"])
    out["p_adj"] = bh_adjust(out["p"].values)
    return out


# -------------------------
# 16) Scale validation: Cronbach alpha + PA scree + AAC + optional EFA
# -------------------------
def impute_mean_df(df_num: pd.DataFrame) -> pd.DataFrame:
    df = df_num.copy()
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        m = float(x.mean(skipna=True)) if np.isfinite(x.mean(skipna=True)) else 0.0
        df[c] = x.fillna(m)
    return df


def cronbach_alpha(df: pd.DataFrame) -> Dict[str, float]:
    X = df.values.astype(float)
    k = X.shape[1]
    if k < 2:
        return {"raw_alpha": np.nan, "std_alpha": np.nan, "average_r": np.nan}

    # raw alpha
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    raw = (k/(k-1)) * (1 - item_var.sum()/total_var) if total_var > 0 else np.nan

    # standardized alpha (use correlation)
    C = np.corrcoef(X, rowvar=False)
    rbar = (C.sum() - k) / (k*(k-1))
    std = (k*rbar) / (1 + (k-1)*rbar) if np.isfinite(rbar) else np.nan

    return {"raw_alpha": float(raw), "std_alpha": float(std), "average_r": float(rbar)}


def compute_aac_top3(eigs: np.ndarray) -> Dict[str, float]:
    eigs = np.array(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs) & (eigs > 0)]
    if eigs.size < 3:
        return {"e1": np.nan, "e2": np.nan, "e3": np.nan, "AAC": np.nan}
    e1, e2, e3 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    r1 = e1/e2
    r2 = e2/e3
    t = (r1/r2)
    aac = t/(1+t)
    return {"e1": e1, "e2": e2, "e3": e3, "AAC": float(aac)}


def save_pa_scree(df_imp: pd.DataFrame, k_guess: int, out_png: str, n_iter: int = 20, seed: int = 123) -> str:
    rng = np.random.default_rng(seed)
    X = df_imp.values.astype(float)
    n, p = X.shape
    C = np.corrcoef(X, rowvar=False)
    eig_obs = np.linalg.eigvalsh(C)[::-1]

    eig_rand = []
    for _ in range(n_iter):
        Z = rng.normal(size=(n, p))
        Cr = np.corrcoef(Z, rowvar=False)
        eig_rand.append(np.linalg.eigvalsh(Cr)[::-1])
    eig_rand = np.mean(np.vstack(eig_rand), axis=0)

    fig = plt.figure(figsize=(7.4, 5.2), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, p+1), eig_obs, marker="o", label="Observed")
    ax.plot(np.arange(1, p+1), eig_rand, marker="o", label="Random mean")
    ax.axvline(k_guess, linestyle="dotted")
    ax.set_title(f"Parallel Analysis (approx) | k_guess={k_guess}")
    ax.set_xlabel("Component / Factor")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return os.path.basename(out_png)


def run_efa_optional(df_imp: pd.DataFrame, k_guess: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not _HAVE_FACTOR_ANALYZER:
        return None, None
    try:
        fa = FactorAnalyzer(n_factors=int(k_guess), rotation="oblimin")
        fa.fit(df_imp.values.astype(float))
        load = pd.DataFrame(fa.loadings_, index=df_imp.columns,
                            columns=[f"F{i+1}" for i in range(int(k_guess))]).reset_index().rename(columns={"index": "Item"})
        # factor_analyzer doesn't always expose Phi; skip if absent
        phi_df = None
        if hasattr(fa, "phi_") and fa.phi_ is not None:
            phi = pd.DataFrame(fa.phi_, columns=[f"F{i+1}" for i in range(int(k_guess))])
            phi.insert(0, "Factor", [f"F{i+1}" for i in range(int(k_guess))])
            phi_df = phi
        return load, phi_df
    except Exception:
        return None, None


# -------------------------
# 17) Baseline helpers
# -------------------------
def build_factor_assignment_from_efa(efa_loadings: Optional[pd.DataFrame], items: List[str], K: int,
                                     loading_floor: float = 0.0) -> Dict[str, Any]:
    def fallback():
        fac = {it: (i % K) + 1 for i, it in enumerate(items)}
        factor_items = {}
        for it, f in fac.items():
            factor_items.setdefault(f, []).append(it)
        factor_label = {}
        for f in range(1, K+1):
            its = factor_items.get(f, [])
            top_it = its[0] if its else ""
            factor_label[f] = f"F{f}_{top_it}" if top_it else f"F{f}_Unnamed"
        return {"item_factor": fac, "factor_items": factor_items, "factor_label": factor_label, "K": K}

    if efa_loadings is None or efa_loadings.empty or "Item" not in efa_loadings.columns:
        return fallback()

    fac_cols = [c for c in efa_loadings.columns if c != "Item"]
    if len(fac_cols) < 2:
        return fallback()

    K2 = min(int(K), len(fac_cols))
    L = efa_loadings.copy()
    L = L[L["Item"].isin(items)].copy()
    if L.empty:
        return fallback()

    mat = L.set_index("Item")[fac_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    mat = np.abs(mat)
    row_items = L["Item"].astype(str).tolist()

    item_factor = {}
    for it in items:
        if it not in row_items:
            item_factor[it] = 1
            continue
        i = row_items.index(it)
        v = mat[i, :K2]
        mx = float(np.max(v)) if v.size else 0.0
        if (not np.isfinite(mx)) or (mx < loading_floor):
            item_factor[it] = 1
        else:
            item_factor[it] = int(np.argmax(v) + 1)

    factor_items = {}
    for it, f in item_factor.items():
        factor_items.setdefault(f, []).append(it)

    factor_label = {}
    for f in range(1, K2+1):
        its = factor_items.get(f, [])
        if not its:
            factor_label[f] = f"F{f}_Unnamed"
            continue
        # pick top item by loading on that factor
        # (approx: use mat and row index)
        scores = []
        for it in its:
            i = row_items.index(it) if it in row_items else None
            sc = float(mat[i, f-1]) if i is not None else 0.0
            scores.append((sc, it))
        scores.sort(key=lambda x: (-x[0], x[1]))
        factor_label[f] = f"F{f}_{scores[0][1]}" if scores else f"F{f}_Unnamed"

    return {"item_factor": item_factor, "factor_items": factor_items, "factor_label": factor_label, "K": K2}


def compute_factor_sums(df_num: pd.DataFrame, item_factor: Dict[str, int], K: int, scale_items: bool = True) -> pd.DataFrame:
    df_imp = impute_mean_df(df_num)
    X = df_imp.values.astype(float)
    if scale_items:
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    ids = df_num.index.astype(str).tolist()
    out = pd.DataFrame({"ID": ids})
    cols = df_num.columns.tolist()
    for f in range(1, int(K)+1):
        its = [it for it in cols if item_factor.get(it, 1) == f]
        coln = f"Factor{f}_Sum"
        if not its:
            out[coln] = 0.0
        else:
            idx = [cols.index(it) for it in its]
            out[coln] = X[:, idx].mean(axis=1)
    out["total_score"] = df_imp.values.astype(float).mean(axis=1)
    return out


import numpy as np
import pandas as pd

def compute_silhouette_euclid(X, cluster, max_exact_n: int = 600):
    """
    Returns:
      sil_df (pd.DataFrame): columns a_i, b_i, SS_i, a_star for rows kept by okmask
      okmask (np.ndarray[bool]): mask over the ORIGINAL rows in X/cluster
    """
    X = np.asarray(X, dtype=float)
    cluster = np.asarray(cluster)

    okmask = np.isfinite(cluster) & (cluster.astype(int) > 0)
    X2 = X[okmask, :]
    cl = cluster[okmask].astype(int)

    n = X2.shape[0]
    uniq = np.unique(cl)

    # Degenerate cases
    if n < 3 or uniq.size < 2:
        a_i = np.zeros(n, dtype=float)
        b_i = np.zeros(n, dtype=float)
        ss  = np.zeros(n, dtype=float)
        a_star = np.ones(n, dtype=float)
        sil_df = pd.DataFrame({"a_i": a_i, "b_i": b_i, "SS_i": ss, "a_star": a_star})
        return sil_df, okmask

    # Exact pairwise distances if not too big (O(n^2))
    if n <= max_exact_n:
        # D[i,j] = ||X2[i]-X2[j]||
        diff = X2[:, None, :] - X2[None, :, :]
        D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

        a_i = np.zeros(n, dtype=float)
        b_i = np.zeros(n, dtype=float)

        idx_all = np.arange(n)
        for i in range(n):
            same = idx_all[(cl == cl[i]) & (idx_all != i)]
            a_i[i] = float(D[i, same].mean()) if same.size else 0.0

            other_means = []
            for c in uniq:
                if c == cl[i]:
                    continue
                other_means.append(float(D[i, cl == c].mean()))
            b_i[i] = float(np.min(other_means)) if other_means else 0.0

    # Approximate for large n using distances-to-centroids (O(nk))
    else:
        centroids = {c: X2[cl == c].mean(axis=0) for c in uniq}
        a_i = np.array([np.linalg.norm(X2[i] - centroids[cl[i]]) for i in range(n)], dtype=float)

        b_i = np.empty(n, dtype=float)
        for i in range(n):
            others = [np.linalg.norm(X2[i] - centroids[c]) for c in uniq if c != cl[i]]
            b_i[i] = float(np.min(others)) if others else 0.0

    den = np.maximum(a_i, b_i)
    ss = np.where(den == 0, 0.0, (b_i - a_i) / den)
    a_star = 1.0 / (1.0 + a_i)

    sil_df = pd.DataFrame({"a_i": a_i, "b_i": b_i, "SS_i": ss, "a_star": a_star})
    return sil_df, okmask



# -------------------------
# 18) HTML helpers + heatmap
# -------------------------
def html_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def html_table(df: Optional[pd.DataFrame], caption: Optional[str] = None, max_rows: int = 80, digits: int = 4) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='small'>(empty)</p>"
    d = df.copy().head(int(max_rows))
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(int(digits))
    cap = f"<div class='cap'>{html_escape(caption)}</div>" if caption else ""
    th = "<tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in d.columns) + "</tr>"
    rows = []
    for _, r in d.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html_escape('' if pd.isna(v) else v)}</td>" for v in r.values) + "</tr>")
    return cap + "<table class='tbl'><thead>" + th + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def safe_img(src: Optional[str]) -> str:
    if not src:
        return "<p class='small'>(no image)</p>"
    return f"<img src='{html_escape(src)}' />"


def val_to_color(x: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(x) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return "#ffffff"
    t = (x - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, float(t)))
    # simple blue-white-red ramp
    # blue: (43,108,176)  white:(255,255,255) red:(197,48,48)
    if t < 0.5:
        tt = t/0.5
        r = int(43 + (255-43)*tt)
        g = int(108 + (255-108)*tt)
        b = int(176 + (255-176)*tt)
    else:
        tt = (t-0.5)/0.5
        r = int(255 + (197-255)*tt)
        g = int(255 + (48-255)*tt)
        b = int(255 + (48-255)*tt)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_person_heatmap_top10_html(df_num: pd.DataFrame, theme_df: pd.DataFrame, p_nodes2: pd.DataFrame,
                                    top_n: int = 10, max_themes: int = 10) -> str:
    df = df_num.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    td = theme_df.copy()
    pn = p_nodes2.copy()
    pn["name"] = pn["name"].astype(str)
    pn["SS_i"] = pd.to_numeric(pn["SS_i"], errors="coerce")

    td["ID"] = td["ID"].astype(str)
    td["theme2_cluster"] = pd.to_numeric(td["theme2_cluster"], errors="coerce").fillna(0).astype(int)
    td["theme2"] = td["theme2"].astype(str)
    td["SS_i"] = td["ID"].map(dict(zip(pn["name"], pn["SS_i"])))

    td = td[(td["theme2_cluster"] > 0)].copy()
    if td.empty:
        return "<p class='small'>(no theme rows for heatmap)</p>"

    counts = td["theme2_cluster"].value_counts()
    keep = counts.index.astype(int).tolist()[:min(len(counts), int(max_themes))]
    blocks = []

    for tc in keep:
        sub = td[td["theme2_cluster"] == tc].copy()
        sub = sub.sort_values(["SS_i", "ID"], ascending=[False, True]).head(int(top_n))
        ids = sub["ID"].tolist()

        mat = df.loc[ids].copy()
        vmin = float(np.nanmin(mat.values)) if np.isfinite(np.nanmin(mat.values)) else 0.0
        vmax = float(np.nanmax(mat.values)) if np.isfinite(np.nanmax(mat.values)) else 1.0

        head_tr = "<tr><th>ID</th><th>SS(i)</th>" + "".join(f"<th>{html_escape(c)}</th>" for c in mat.columns) + "</tr>"
        body_rows = []
        for rid in mat.index.astype(str).tolist():
            ss = float(sub.loc[sub["ID"] == rid, "SS_i"].values[0]) if (sub["ID"] == rid).any() else np.nan
            tds = []
            for c in mat.columns:
                x = float(mat.loc[rid, c]) if np.isfinite(mat.loc[rid, c]) else np.nan
                col = val_to_color(x, vmin, vmax)
                tds.append(f"<td style='background:{col};' title='{html_escape(f'{x:.4f}' if np.isfinite(x) else '')}'></td>")
            body_rows.append(
                f"<tr><td class='group'>{html_escape(rid)}</td><td>{html_escape(f'{ss:.4f}' if np.isfinite(ss) else '')}</td>"
                + "".join(tds) + "</tr>"
            )

        blocks.append(
            f"<div class='cap'>Heatmap (Top {top_n} SS(i)) — theme2_cluster={tc} ({html_escape(sub['theme2'].iloc[0])})</div>"
            + "<table class='tbl'><thead>" + head_tr + "</thead><tbody>" + "".join(body_rows) + "</tbody></table>"
        )

    return "<div class='card'><h3>PERSON Heatmap (color only)</h3>" + "<hr/>".join(blocks) + "</div>"


# -------------------------
# 19) Pick best mode
# -------------------------
def pick_best_mode(summary_df: pd.DataFrame) -> str:
    df = summary_df.copy()
    for nm in ["kappa_theme_vs_label", "auc_true", "term_overall_SS", "modularity_Q"]:
        if nm not in df.columns:
            df[nm] = np.nan
    df["kappa"] = pd.to_numeric(df["kappa_theme_vs_label"], errors="coerce")
    df["auc_true"] = pd.to_numeric(df["auc_true"], errors="coerce")
    df["termSS"] = pd.to_numeric(df["term_overall_SS"], errors="coerce")
    df["Q"] = pd.to_numeric(df["modularity_Q"], errors="coerce")

    # sort: kappa desc, auc_true desc, termSS desc, Q desc, mode asc
    df = df.sort_values(
        by=["kappa", "auc_true", "termSS", "Q", "mode"],
        ascending=[False, False, False, False, True],
        na_position="last"
    )
    return str(df["mode"].iloc[0])


# -------------------------
# 20) One mode runner (A/B/C)
# -------------------------
def modularity_Q_from_partition(edges: pd.DataFrame, node_cluster: Dict[str, int]) -> float:
    if edges is None or edges.empty or not node_cluster:
        return float("nan")
    G = nx.Graph()
    for n, c in node_cluster.items():
        G.add_node(str(n))
    for _, r in edges.iterrows():
        a = str(r["term1"]); b = str(r["term2"]); w = float(r["wcd"])
        if a == b:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)

    # build communities list
    comm_map: Dict[int, set] = {}
    for n, c in node_cluster.items():
        comm_map.setdefault(int(c), set()).add(str(n))
    communities = list(comm_map.values())
    try:
        from networkx.algorithms.community.quality import modularity
        return float(modularity(G, communities, weight="weight"))
    except Exception:
        return float("nan")


def run_one_mode(mode: str,
                 df_num: pd.DataFrame,
                 df_raw: pd.DataFrame,
                 true_label: np.ndarray,
                 target_clusters: int,
                 u_y: List[int],
                 min_wcd: int = 1,
                 top_k: int = 20,
                 verbose: bool = True) -> Dict[str, Any]:
    tokens_list, thresholds = make_tokens_YN(df_num, mode=mode, thr_method="mean", na_as_zero=True)
    nodes_all, edges_all = build_coword(tokens_list, min_wcd=min_wcd)
    if nodes_all.empty:
        raise RuntimeError("No tokens produced.")

    fl_nodes, rel_one = flca_cluster_with_onelink(nodes_all, edges_all, target_clusters=target_clusters, verbose=verbose)
    nodes_all2 = compute_silhouette_graph(fl_nodes, edges_all)

    # modularity
    node_cluster_map = dict(zip(nodes_all2["name"].astype(str), nodes_all2["cluster"].astype(int)))
    Q = modularity_Q_from_partition(edges_all, node_cluster_map)

    wv = pd.to_numeric(nodes_all2["value"], errors="coerce").fillna(1).values.astype(float)
    ss = pd.to_numeric(nodes_all2["SS_i"], errors="coerce").fillna(0).values.astype(float)
    term_overall_ss = float((wv*ss).sum() / max(1.0, wv.sum()))

    clusters = sorted(nodes_all2["cluster"].unique().tolist())
    cluster_leader_map = {}
    for c in clusters:
        sub = nodes_all2[nodes_all2["cluster"] == c].copy()
        sub = sub.sort_values(["value", "name"], ascending=[False, True])
        cluster_leader_map[int(c)] = str(sub["name"].iloc[0])

    K = len(clusters)

    # top-k
    top_nodes = majority_topk(nodes_all2, k=int(top_k))
    nodes20 = nodes_all2[nodes_all2["name"].astype(str).isin(top_nodes)].copy()
    nodes20 = nodes20.sort_values(["cluster", "value", "name"], ascending=[True, False, True])
    rel20 = rel_one[(rel_one["follower"].isin(top_nodes)) & (rel_one["leader"].isin(top_nodes))].copy()

    # TAAA
    taaa_df = taaa_rounds(tokens_list, node_cluster_map=node_cluster_map, K=K,
                          cluster_leader_map=cluster_leader_map)

    taaa_show = taaa_df.copy()
    # attach TRUE_LABEL from training rows
    taaa_show["TRUE_LABEL"] = taaa_show["ID"].map(dict(zip(df_raw.index.astype(str), true_label)))
    taaa_show = taaa_show[["ID", "TRUE_LABEL", "theme1_cluster", "theme1", "support1", "theme2_cluster", "theme2"]]

    r2 = eval_kxk(true_label, taaa_df["theme2_cluster"].values, K=K)

    # TERM Kano
    term_kano_title = f"TERM Kano (Mode {mode}) | Kappa(R2)={r2['kappa']:.3f} | SS={term_overall_ss:.3f} | Q={Q:.3f}"
    term_kano_png = os.path.join(FIG_DIR, f"term_kano_mode_{mode}.png")
    term_kano_path = os.path.join("figures", save_kano_term(nodes_all2, rel_one, term_kano_title, term_kano_png))

    # TERM network top-k (optional)
    term_net_path = None
    if not rel20.empty and len(nodes20) >= 2:
        Gd = nx.DiGraph()
        for _, r in nodes20.iterrows():
            Gd.add_node(str(r["name"]), value=float(r["value"]), cluster=int(r["cluster"]))
        for _, r in rel20.iterrows():
            Gd.add_edge(str(r["follower"]), str(r["leader"]), w=float(r.get("wcd", 1)))
        pos = nx.spring_layout(Gd.to_undirected(), seed=123)

        fig = plt.figure(figsize=(10, 6), dpi=160)
        ax = fig.add_subplot(111)
        for a, b, d in Gd.edges(data=True):
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], alpha=0.35, linewidth=max(0.6, d.get("w", 1)/4.0))
        xs = [pos[n][0] for n in Gd.nodes()]
        ys = [pos[n][1] for n in Gd.nodes()]
        sizes = [25 + 3*np.sqrt(Gd.nodes[n].get("value", 1.0)) for n in Gd.nodes()]
        ax.scatter(xs, ys, s=sizes, alpha=0.9)
        for n in Gd.nodes():
            ax.text(pos[n][0], pos[n][1], n, fontsize=8, fontweight="bold", ha="center", va="bottom")
        ax.set_title(f"TERM ONE-LINK Network Top-{top_k} (Mode {mode})")
        ax.axis("off")
        fig.tight_layout()
        png = os.path.join(FIG_DIR, f"term_onelink_net_mode_{mode}.png")
        fig.savefig(png)
        plt.close(fig)
        term_net_path = os.path.join("figures", os.path.basename(png))

    # PERSON graph (from tokens)
    p_nodes, p_edges = build_person_graph_from_tokens(tokens_list, min_shared=2, top_m=8)
    p_fl_nodes, _ = flca_cluster_with_onelink(p_nodes, p_edges, target_clusters=target_clusters, verbose=verbose)
    p_nodes2 = compute_silhouette_graph(p_fl_nodes, p_edges)

    # attach theme2 for coloring
    theme_map = dict(zip(taaa_df["ID"].astype(str), taaa_df["theme2_cluster"].astype(int)))
    theme_name_map = dict(zip(taaa_df["ID"].astype(str), taaa_df["theme2"].astype(str)))
    p_nodes2["theme2_cluster"] = p_nodes2["name"].astype(str).map(theme_map).fillna(0).astype(int)
    p_nodes2["theme2"] = p_nodes2["name"].astype(str).map(theme_name_map).fillna("Unassigned")

    pv = pd.to_numeric(p_nodes2["value"], errors="coerce").fillna(1).values.astype(float)
    pss = pd.to_numeric(p_nodes2["SS_i"], errors="coerce").fillna(0).values.astype(float)
    person_overall_ss = float((pv*pss).sum() / max(1.0, pv.sum()))

    # PERSON Kano (no dummy overlay here; done later for BEST mode only)
    person_kano_title = f"PERSON Kano (Mode {mode}) — colored by theme2_cluster"
    person_kano_png = os.path.join(FIG_DIR, f"person_kano_mode_{mode}.png")
    person_kano_path = os.path.join("figures", save_kano_person(p_nodes2, person_kano_title, "theme2_cluster", person_kano_png))

    # ROC + Logistic regression (binary only)
    roc_path = None
    roc_tbl = None
    logit_tbl = None
    if len(u_y) == 2:
        y_true01 = (true_label == max(u_y)).astype(int)
        # align to p_nodes2 order
        y_true01_p = p_nodes2["name"].astype(str).map(dict(zip(df_raw.index.astype(str), y_true01))).values
        # choose internal positive cluster (most TRUE=1 proportion)
        tab_ct = pd.crosstab(p_nodes2["cluster"], y_true01_p)
        cluster_pos = int(tab_ct.index[0]) if len(tab_ct.index) else 1
        if 1 in tab_ct.columns:
            prop1 = (tab_ct[1] / (tab_ct.sum(axis=1) + 1e-12)).values
            cluster_pos = int(tab_ct.index[int(np.argmax(prop1))])

        y_internal01 = (pd.to_numeric(p_nodes2["cluster"], errors="coerce").fillna(0).astype(int).values == cluster_pos).astype(int)

        roc_internal = roc_compute(p_nodes2["SS_i"].values, y_internal01, n_boot=200, seed=123)
        roc_true = roc_compute(p_nodes2["SS_i"].values, y_true01_p, n_boot=200, seed=123)

        roc_png = os.path.join(FIG_DIR, f"roc_combined_mode_{mode}.png")
        roc_rel = plot_roc_combined(roc_internal, roc_true, f"Combined ROC (Mode {mode}): person SS(i)", roc_png)
        roc_path = os.path.join("figures", roc_rel) if roc_rel else None

        def best_row(roc):
            b = roc.get("best")
            return b if isinstance(b, dict) else {}

        bi = best_row(roc_internal)
        bt = best_row(roc_true)
        roc_tbl = pd.DataFrame({
            "Target": [f"Internal ROC (circular) (cluster={cluster_pos})", "TRUE_LABEL"],
            "AUC": [roc_internal["auc"], roc_true["auc"]],
            "CI_low": [roc_internal["ci"][0], roc_true["ci"][0]],
            "CI_high": [roc_internal["ci"][1], roc_true["ci"][1]],
            "Cutpoint": [bi.get("thr", np.nan), bt.get("thr", np.nan)],
            "Sens": [bi.get("sens", np.nan), bt.get("sens", np.nan)],
            "Spec": [bi.get("spec", np.nan), bt.get("spec", np.nan)]
        })

        th1 = p_nodes2["name"].astype(str).map(dict(zip(taaa_df["ID"].astype(str), taaa_df["theme1_cluster"].astype(int)))).values
        th2 = p_nodes2["name"].astype(str).map(dict(zip(taaa_df["ID"].astype(str), taaa_df["theme2_cluster"].astype(int)))).values

        logit_tbl = pd.concat([
            logistic_ssi_table(p_nodes2["SS_i"].values, y_true01_p, "TRUE_LABEL ~ SS(i)"),
            logistic_ssi_to_theme_ovr_table(p_nodes2["SS_i"].values, th2, "TAAA2 theme2_cluster ~ SS(i)"),
            logistic_ssi_to_theme_ovr_table(p_nodes2["SS_i"].values, th1, "TAAA1 theme1_cluster ~ SS(i)"),
            logistic_ssi_to_theme_ovr_table(p_nodes2["SS_i"].values, p_nodes2["cluster"].values, "PERSON_FLCA_cluster ~ SS(i)")
        ], ignore_index=True)

    # testlets + UPDATED paired ANOVA tables
    testlet_obj = build_item_testlets_from_term_nodes(nodes_all2)
    testlet_scores = compute_testlet_scores(df_num, testlet_obj)
    testlet_scores["theme2_cluster"] = testlet_scores["ID"].map(dict(zip(taaa_df["ID"].astype(str), taaa_df["theme2_cluster"].astype(int))))
    testlet_scores["theme2"] = testlet_scores["ID"].map(dict(zip(taaa_df["ID"].astype(str), taaa_df["theme2"].astype(str))))
    testlet_scores["TRUE_LABEL"] = testlet_scores["ID"].map(dict(zip(df_raw.index.astype(str), true_label)))

    anova_by_theme = build_anova_table_testlets(testlet_scores, group_col="theme2_cluster", testlet_obj=testlet_obj)
    anova_by_true = build_anova_table_testlets(testlet_scores, group_col="TRUE_LABEL", testlet_obj=testlet_obj)

    heatmap_html = build_person_heatmap_top10_html(
        df_num=df_num,
        theme_df=taaa_show[["ID", "theme2_cluster", "theme2", "TRUE_LABEL"]],
        p_nodes2=p_nodes2,
        top_n=10,
        max_themes=10
    )

    return {
        "mode": mode,
        "K": K,
        "term": {
            "overall_ss": term_overall_ss,
            "Q": Q,
            "kano_path": term_kano_path,
            "net_path": term_net_path,
            "nodes_tbl": nodes20[["name", "cluster", "value", "Leader", "a_i", "b_i", "a_star", "SS_i"]].copy(),
            "rel_tbl": rel_one.head(80).copy(),
            "taaa_show": taaa_show.copy(),
            "kappa": {
                "value": float(r2["kappa"]),
                "tab_mapped": pd.DataFrame(r2["tab_mapped"],
                                          index=[f"TRUTH{i+1}" for i in range(K)],
                                          columns=[f"PRED{i+1}" for i in range(K)]),
                "n_used": int(r2["n_used"])
            }
        },
        "person": {
            "overall_ss": person_overall_ss,
            "kano_path": person_kano_path,
            "kano_pred_path": None,
            "pred_table": None,
            "roc_path": roc_path,
            "roc_tbl": roc_tbl,
            "logit_tbl": logit_tbl,
            "heatmap_html": heatmap_html,
            "testlet_scores": testlet_scores,
            "anova_by_theme": anova_by_theme,
            "anova_by_true": anova_by_true
        },
        "internals": {
            "thresholds": thresholds,
            "tokens_train": tokens_list,
            "node_cluster_map": node_cluster_map,
            "cluster_leader_map": cluster_leader_map
        }
    }


# -------------------------
# 21) BASELINE modes
# -------------------------
def run_baseline_mode(method: str,
                      df_num: pd.DataFrame,
                      df_raw: pd.DataFrame,
                      true_label: np.ndarray,
                      target_clusters: int,
                      factor_assign: Dict[str, Any],
                      factor_sums_df: pd.DataFrame) -> Dict[str, Any]:
    K = int(factor_assign["K"])
    ids = factor_sums_df["ID"].astype(str).tolist()

    # baseline "testlets" = EFA factor groups
    testlet_items = factor_assign.get("factor_items", {})
    testlet_items = {int(k): v for k, v in testlet_items.items()}
    testlet_obj = {"testlet_items": testlet_items}

    if method in ("ARGMAX", "KMEANS"):
        fac_cols = [c for c in factor_sums_df.columns if c.startswith("Factor") and c.endswith("_Sum")]
        fac_cols = fac_cols[:K]
        X = factor_sums_df[fac_cols].values.astype(float)
        X[~np.isfinite(X)] = 0.0
    else:
        aug_df = compute_testlet_scores_augmax(df_num, testlet_obj)
        tcols = sorted([c for c in aug_df.columns if c.startswith("Testlet") and c[7:].isdigit()])
        X = aug_df[tcols].values.astype(float) if tcols else np.zeros((len(aug_df), K), dtype=float)
        X[~np.isfinite(X)] = 0.0

    if method in ("ARGMAX", "AUGMAX"):
        pred = np.argmax(X, axis=1) + 1
    else:
        if not _HAVE_SKLEARN:
            # fallback: random-ish stable split if sklearn missing
            pred = (np.argmax(X, axis=1) + 1).astype(int)
        else:
            k = min(K, X.shape[0])
            k = max(2, k)
            km = KMeans(n_clusters=k, n_init=25, max_iter=50, random_state=123)
            pred = km.fit_predict((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)) + 1

    pred = pred.astype(int)
    pred[pred < 1] = 1

    sil_df, okmask = compute_silhouette_euclid(X, pred)
    id2 = np.array(ids)[okmask]
    pred2 = pred[okmask]

    ts_map = dict(zip(factor_sums_df["ID"].astype(str), pd.to_numeric(factor_sums_df["total_score"], errors="coerce").fillna(0).values))
    ts = np.array([ts_map.get(i, 0.0) for i in id2], dtype=float)

    # leader per cluster by total_score
    leader_by_cluster = {}
    for cc in sorted(set(pred2.tolist())):
        idx = np.where(pred2 == cc)[0]
        if idx.size:
            leader_by_cluster[cc] = str(id2[idx[np.argmax(ts[idx])]])
    Leader = np.array([leader_by_cluster.get(int(cc), str(id2[i])) for i, cc in enumerate(pred2)])

    # theme labels
    factor_label = factor_assign.get("factor_label", {})
    theme2 = np.array([factor_label.get(int(cc), "F_Unnamed") for cc in pred2], dtype=object)

    p_nodes2 = pd.DataFrame({
        "name": id2,
        "cluster": pred2,
        "theme2_cluster": pred2,
        "theme2": theme2,
        "value": ts,
        "Leader": Leader,
        "a_i": sil_df["a_i"].values,
        "b_i": sil_df["b_i"].values,
        "a_star": sil_df["a_star"].values,
        "SS_i": sil_df["SS_i"].values
    })

    pv = pd.to_numeric(p_nodes2["value"], errors="coerce").fillna(1).values.astype(float)
    pss = pd.to_numeric(p_nodes2["SS_i"], errors="coerce").fillna(0).values.astype(float)
    person_overall_ss = float((pv*pss).sum() / max(1.0, pv.sum()))

    mode_name = method if method not in ("KMEANS", "KMEANS_AUGMAX") else f"{method}{K}"
    kano_png = os.path.join(FIG_DIR, f"person_kano_{mode_name}.png")
    kano_path = os.path.join("figures", save_kano_person(p_nodes2, f"PERSON Kano (BASELINE {mode_name}) — colored by theme_cluster",
                                                         "theme2_cluster", kano_png))

    theme_df = pd.DataFrame({
        "ID": id2,
        "TRUE_LABEL": pd.Series(id2).map(dict(zip(df_raw.index.astype(str), true_label))).values,
        "theme1_cluster": pred2,
        "theme1": theme2,
        "support1": np.nan,
        "theme2_cluster": pred2,
        "theme2": theme2
    })

    r2 = eval_kxk(pd.Series(id2).map(dict(zip(df_raw.index.astype(str), true_label))).values, pred2, K=K)

    # ANOVA + heatmap
    testlet_scores = compute_testlet_scores(df_num, testlet_obj)
    pred_map = dict(zip(ids, pred))
    name_map = dict(zip(id2, theme2))
    testlet_scores["theme2_cluster"] = testlet_scores["ID"].map(pred_map)
    testlet_scores["theme2"] = testlet_scores["ID"].map(name_map)
    testlet_scores["TRUE_LABEL"] = testlet_scores["ID"].map(dict(zip(df_raw.index.astype(str), true_label)))

    anova_by_theme = build_anova_table_testlets(testlet_scores, group_col="theme2_cluster", testlet_obj=testlet_obj)
    anova_by_true = build_anova_table_testlets(testlet_scores, group_col="TRUE_LABEL", testlet_obj=testlet_obj)

    heatmap_html = build_person_heatmap_top10_html(
        df_num=df_num,
        theme_df=theme_df[["ID", "theme2_cluster", "theme2", "TRUE_LABEL"]],
        p_nodes2=p_nodes2,
        top_n=10, max_themes=10
    )

    return {
        "mode": mode_name,
        "K": K,
        "term": {
            "overall_ss": np.nan,
            "Q": np.nan,
            "kano_path": None,
            "net_path": None,
            "nodes_tbl": pd.DataFrame(),
            "rel_tbl": pd.DataFrame(),
            "taaa_show": theme_df[["ID", "TRUE_LABEL", "theme1_cluster", "theme1", "support1", "theme2_cluster", "theme2"]].copy(),
            "kappa": {
                "value": float(r2["kappa"]),
                "tab_mapped": pd.DataFrame(r2["tab_mapped"],
                                          index=[f"TRUTH{i+1}" for i in range(K)],
                                          columns=[f"PRED{i+1}" for i in range(K)]),
                "n_used": int(r2["n_used"])
            }
        },
        "person": {
            "overall_ss": person_overall_ss,
            "kano_path": kano_path,
            "kano_pred_path": None,
            "pred_table": None,
            "roc_path": None,
            "roc_tbl": None,
            "logit_tbl": None,
            "heatmap_html": heatmap_html,
            "testlet_scores": testlet_scores,
            "anova_by_theme": anova_by_theme,
            "anova_by_true": anova_by_true
        },
        "internals": {
            "thresholds": None, "tokens_train": None, "node_cluster_map": None, "cluster_leader_map": None
        }
    }


# -------------------------
# NEW: Add pred patient overlay to BEST mode (token modes only)
# -------------------------
def predict_theme_from_tokens(tokens: List[str],
                             node_cluster_map: Dict[str, int],
                             cluster_leader_map: Dict[int, str]) -> Dict[str, Any]:
    if not tokens:
        return {"theme2_cluster": 0, "theme2": "Unassigned", "support": 0}
    toks = [t for t in tokens if t in node_cluster_map]
    if not toks:
        return {"theme2_cluster": 0, "theme2": "Unassigned", "support": 0}
    cc = [int(node_cluster_map[t]) for t in toks]
    tb = {}
    for c in cc:
        tb[c] = tb.get(c, 0) + 1
    mx = max(tb.values())
    cand = [c for c, v in tb.items() if v == mx]
    th = min(cand)
    th_name = cluster_leader_map.get(th, f"Theme_{th}")
    return {"theme2_cluster": th, "theme2": th_name, "support": int(mx)}


def add_pred_patient_overlay_to_mode(x_mode: Dict[str, Any],
                                    df_pred_patient: pd.DataFrame,
                                    df_num_cols: List[str],
                                    target_clusters: int) -> Dict[str, Any]:
    if df_pred_patient is None or df_pred_patient.empty or len(df_pred_patient) != 1:
        return x_mode
    if not x_mode.get("internals") or x_mode["internals"].get("thresholds") is None:
        # baseline modes: skip
        return x_mode

    thresholds: pd.Series = x_mode["internals"]["thresholds"]
    mode = x_mode["mode"]
    node_cluster_map = x_mode["internals"]["node_cluster_map"]
    cluster_leader_map = x_mode["internals"]["cluster_leader_map"]
    tokens_train = x_mode["internals"]["tokens_train"]

    one_row = df_pred_patient.iloc[0].reindex(df_num_cols)
    one_row = pd.to_numeric(one_row, errors="coerce")
    tokens_pred = tokenize_one_row_YN(one_row, thresholds=thresholds, mode=mode, na_as_zero=True)
    pred_theme = predict_theme_from_tokens(tokens_pred, node_cluster_map, cluster_leader_map)

    tokens_all = dict(tokens_train)
    tokens_all[PRED_PATIENT_ID] = sorted(set(tokens_pred))

    pg_nodes, pg_edges = build_person_graph_from_tokens(tokens_all, min_shared=2, top_m=8)
    p_fl_nodes, _ = flca_cluster_with_onelink(pg_nodes, pg_edges, target_clusters=target_clusters, verbose=False)
    p_nodes2_all = compute_silhouette_graph(p_fl_nodes, pg_edges)

    # attach theme for coloring
    theme_df = x_mode["term"]["taaa_show"][["ID", "theme2_cluster", "theme2"]].copy()
    theme_map = dict(zip(theme_df["ID"].astype(str), pd.to_numeric(theme_df["theme2_cluster"], errors="coerce").fillna(0).astype(int)))
    theme_name_map = dict(zip(theme_df["ID"].astype(str), theme_df["theme2"].astype(str)))

    p_nodes2_all["theme2_cluster"] = p_nodes2_all["name"].astype(str).map(theme_map).fillna(0).astype(int)
    p_nodes2_all["theme2"] = p_nodes2_all["name"].astype(str).map(theme_name_map).fillna("Unassigned")

    # overwrite dummy theme with prediction
    p_nodes2_all.loc[p_nodes2_all["name"].astype(str) == PRED_PATIENT_ID, "theme2_cluster"] = int(pred_theme["theme2_cluster"])
    p_nodes2_all.loc[p_nodes2_all["name"].astype(str) == PRED_PATIENT_ID, "theme2"] = str(pred_theme["theme2"])

    png = os.path.join(FIG_DIR, f"person_kano_with_pred_BEST_{mode}.png")
    rel = save_kano_person(
        p_nodes2_all,
        f"PERSON Kano + Pred patient (BEST mode {mode}) — colored by theme2_cluster",
        "theme2_cluster",
        png,
        highlight_name=PRED_PATIENT_ID
    )

    x_mode["person"]["kano_pred_path"] = os.path.join("figures", rel)
    x_mode["person"]["pred_table"] = pd.DataFrame([{
        "ID": PRED_PATIENT_ID,
        "pred_theme2_cluster": int(pred_theme["theme2_cluster"]),
        "pred_theme2": str(pred_theme["theme2"]),
        "support": int(pred_theme["support"])
    }])
    return x_mode


# -------------------------
# 22) HTML report builder
# -------------------------
def build_scale_block_html(sb: Dict[str, Any]) -> str:
    if not sb:
        return "<div class='card'><h2>Scale validation</h2><p class='small'>(unavailable)</p></div>"
    parts = ["<div class='card'><h2>Scale validation / Reliability / Convergent validity</h2>"]
    parts.append("<h3>Reliability</h3>")
    parts.append(html_table(sb.get("alpha_tbl"), "Cronbach alpha (available fields only)", max_rows=10, digits=6))
    parts.append("<h3>Parallel Analysis Scree Plot</h3>")
    parts.append(safe_img(sb.get("pa_path")))
    parts.append(html_table(sb.get("aac_tbl"), "AAC (top-3 eigenvalues) from input correlation matrix", max_rows=5, digits=6))
    parts.append(html_table(sb.get("eig_tbl"), "Eigenvalues (top 20)", max_rows=25, digits=6))
    parts.append("<h3>EFA rotated loadings (oblimin) — optional</h3>")
    if sb.get("efa_loadings") is not None:
        parts.append(html_table(sb.get("efa_loadings"), "Loadings (items × factors)", max_rows=120, digits=4))
    else:
        parts.append("<p class='small'>(EFA unavailable: install factor_analyzer to enable)</p>")
    if sb.get("efa_phi") is not None:
        parts.append(html_table(sb.get("efa_phi"), "Factor correlations (Phi)", max_rows=30, digits=4))
    parts.append("</div>")
    return "".join(parts)


def mode_summary_table_html(x: Dict[str, Any]) -> str:
    auc_true = np.nan
    auc_internal = np.nan
    roc_tbl = x["person"].get("roc_tbl")
    if isinstance(roc_tbl, pd.DataFrame) and len(roc_tbl) >= 2:
        auc_internal = float(pd.to_numeric(roc_tbl["AUC"].iloc[0], errors="coerce"))
        auc_true = float(pd.to_numeric(roc_tbl["AUC"].iloc[1], errors="coerce"))

    df = pd.DataFrame({
        "Metric": ["K (clusters)", "Kappa(theme2 vs TRUE_LABEL)", "AUC(TRUE_LABEL) [from SS(i)]",
                   "Internal ROC (circular) [from SS(i)]", "TERM overall SS(i) (weighted)",
                   "TERM modularity Q", "PERSON overall SS(i) (weighted)"],
        "Value": [x["K"], x["term"]["kappa"]["value"], auc_true, auc_internal,
                  x["term"]["overall_ss"], x["term"]["Q"], x["person"]["overall_ss"]]
    })
    return html_table(df, "Mode summary (shown BEFORE plots/tables)", max_rows=30, digits=4)


def mode_card_html(x: Dict[str, Any], is_best: bool) -> str:
    best_tag = "<div class='small'><b>✅ BEST MODE</b></div>" if is_best else ""
    tab = x["term"]["kappa"]["tab_mapped"]
    tab2 = tab.copy()
    tab2.insert(0, "TRUTH", tab2.index.astype(str))

    has_term = bool(x["term"].get("kano_path"))
    parts = [f"<div class='card'>{best_tag}<h2>Mode {html_escape(x['mode'])}</h2>",
             mode_summary_table_html(x),
             "<h3>TERM Kano (with post-FLCA one-link edges)</h3>",
             safe_img(x["term"]["kano_path"]) if has_term else "<p class='small'>(baseline mode: no TERM token network)</p>",
             "<h3>TERM one-link network (Top-k)</h3>",
             safe_img(x["term"]["net_path"]) if x["term"].get("net_path") else "<p class='small'>(no term network)</p>",
             "<h3>TERM nodes (Top-k)</h3>",
             html_table(x["term"]["nodes_tbl"], "Top-k token nodes with a(i), b(i), SS(i), a*", max_rows=40, digits=4),
             "<h3>TERM one-link relations (head)</h3>",
             html_table(x["term"]["rel_tbl"], "Unique follower edge", max_rows=80, digits=0),
             "<h3>Theme assignment table (TAAA or Baseline)</h3>",
             html_table(x["term"]["taaa_show"].head(50), "ID + TRUE_LABEL + theme1/theme2", max_rows=50, digits=0),
             "<h3>Kappa confusion table (mapped)</h3>",
             html_table(tab2, "TRUE_LABEL × mapped theme2_cluster", max_rows=50, digits=0),
             "<h3>PERSON Kano (colored by theme2_cluster)</h3>",
             safe_img(x["person"]["kano_path"])]

    if x["person"].get("kano_pred_path"):
        parts += [
            "<h3>BEST-mode overlay: Pred patient on PERSON Kano (highlighted in red)</h3>",
            safe_img(x["person"]["kano_pred_path"]),
        ]
        if isinstance(x["person"].get("pred_table"), pd.DataFrame):
            parts.append(html_table(x["person"]["pred_table"], "Predicted theme for dummy patient", max_rows=10, digits=4))

    parts += [
        "<h3>ROC (SS(i))</h3>",
        safe_img(x["person"]["roc_path"]) if x["person"].get("roc_path") else "<p class='small'>(ROC not computed)</p>",
        html_table(x["person"]["roc_tbl"], "ROC summary", max_rows=10, digits=4) if isinstance(x["person"].get("roc_tbl"), pd.DataFrame) else "",
        "<h3>Logistic regression (with Wald)</h3>",
        html_table(x["person"]["logit_tbl"], "Beta, SE, z, Wald=z^2, OR, CI, p(Wald), p_LRT", max_rows=220, digits=6) if isinstance(x["person"].get("logit_tbl"), pd.DataFrame) else "<p class='small'>(logit not computed)</p>",
        "<hr/>",
        x["person"]["heatmap_html"],
        "<hr/>",
        "<h3>Testlet scores (df_raw numeric)</h3>",
        html_table(x["person"]["testlet_scores"].head(30), "Head 30: ID + theme2_cluster + TRUE_LABEL + Testlet* means", max_rows=30, digits=4),
        "<h3>Table 5. Within-theme paired test: TestletA vs TestletB</h3>",
        html_table(x["person"]["anova_by_theme"], "Paired within each theme2_cluster (F=t^2), BH-adjusted p", max_rows=200, digits=6),
        "<h3>Table 6. Within-TRUE_LABEL paired test: TestletA vs TestletB</h3>",
        html_table(x["person"]["anova_by_true"], "Paired within each TRUE_LABEL (F=t^2), BH-adjusted p", max_rows=200, digits=6),
        "</div>"
    ]
    return "".join(parts)


def write_report_html(scale_block_html: str, res: Dict[str, Any], out_dir: str,
                      df_num: pd.DataFrame, target_clusters: int, df_pred_patient: Optional[pd.DataFrame]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    modes = list(res.keys())

    summary_rows = []
    for m in modes:
        x = res[m]
        auc_true = np.nan
        auc_internal = np.nan
        roc_tbl = x["person"].get("roc_tbl")
        if isinstance(roc_tbl, pd.DataFrame) and len(roc_tbl) >= 2:
            auc_internal = float(pd.to_numeric(roc_tbl["AUC"].iloc[0], errors="coerce"))
            auc_true = float(pd.to_numeric(roc_tbl["AUC"].iloc[1], errors="coerce"))
        summary_rows.append({
            "mode": m,
            "term_overall_SS": x["term"]["overall_ss"],
            "modularity_Q": x["term"]["Q"],
            "person_overall_SS": x["person"]["overall_ss"],
            "kappa_theme_vs_label": x["term"]["kappa"]["value"],
            "auc_true": auc_true,
            "auc_internal": auc_internal
        })

    summary_df = pd.DataFrame(summary_rows)
    best_mode = pick_best_mode(summary_df)

    # pred overlay ONLY to BEST token-mode (A/B/C) and only if enabled
    if ENABLE_PRED_PATIENT and df_pred_patient is not None and best_mode in res:
        if best_mode in ("A", "B", "C"):
            res[best_mode] = add_pred_patient_overlay_to_mode(
                res[best_mode], df_pred_patient, df_num_cols=df_num.columns.tolist(),
                target_clusters=target_clusters
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    html_ts = os.path.join(out_dir, f"report_{'_'.join(modes)}_{stamp}.html")
    html_latest = os.path.join(out_dir, "report.html")

    glossary = "<ul>" + "".join(f"<li>{html_escape(x)}</li>" for x in GLOSSARY_LINES) + "</ul>"

    css = """
    body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans',Arial; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 14px 0; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
    .small { color:#666; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #eee; margin: 8px 0; }
    .tbl { border-collapse: collapse; width: 100%; margin: 8px 0 6px; }
    .tbl th, .tbl td { border: 1px solid #e6e6e6; padding: 6px 8px; font-size: 13px; }
    .tbl th { background: #fafafa; text-align: left; }
    .cap { font-weight: 700; margin-top: 8px; }
    td.group { font-weight: 700; background: #F9FAFB; }
    hr { border: none; border-top: 1px solid #eee; margin: 18px 0; }
    """

    top_block = (
        "<div class='card'>"
        "<h2>TOP: Global summary + Best mode</h2>"
        f"<div class='small'>Best mode = <b>{html_escape(best_mode)}</b> (rule: kappa → AUC(TRUE_LABEL) → TERM SS → Q)</div>"
        + html_table(summary_df, "Per-mode summary (all modes shown)", max_rows=40, digits=4)
        + "<h3>Glossary</h3>" + glossary
        + (f"<div class='small'>Prediction-only patient ID: <b>{html_escape(PRED_PATIENT_ID)}</b> (excluded from training; overlaid on BEST mode only)</div>"
           if ENABLE_PRED_PATIENT and df_pred_patient is not None else "")
        + "</div>"
    )

    cards = []
    for m in modes:
        cards.append(mode_card_html(res[m], is_best=(m == best_mode)))

    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>TAAA from Digital → Text</title>"
        f"<style>{css}</style></head><body>"
        "<h1>TAAA from Digital → Text (A/B/C) + BASELINES</h1>"
        f"<p class='small'>N={df_num.shape[0]} | Items={df_num.shape[1]} | target_clusters={target_clusters} | Generated={html_escape(time.strftime('%Y-%m-%d %H:%M:%S'))}</p>"
        + scale_block_html
        + top_block
        + "\n".join(cards)
        + "</body></html>"
    )

    # write UTF-8 (with BOM-like effect not needed in HTML; still safe UTF-8)
    with open(html_ts, "w", encoding="utf-8") as f:
        f.write(html)
    with open(html_latest, "w", encoding="utf-8") as f:
        f.write(html)

    return {"latest": html_latest, "timestamp": html_ts, "best_mode": best_mode, "summary": summary_df}


# -------------------------
# MAIN
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="CSV/TSV/TXT/XLSX file path. If omitted, demo data is used.")
    ap.add_argument("--outdir", default=None, help="Output directory. If omitted, uses ./flca_taaa_html_report")
    ap.add_argument("--enable_pred", default=None, help="1/0. If omitted, use code default.")
    ap.add_argument("--pred_json", default=None, help="JSON dict for dummy patient input (optional).")
    args = ap.parse_args()

    # webapp 會用環境變數傳；你也可用 CLI
    input_path = args.input or os.environ.get("INPUT_FILE") or None
    out_dir = args.outdir or os.environ.get("OUT_DIR") or os.path.join(os.getcwd(), "flca_taaa_html_report")
    init_output_dirs(out_dir)

    # 覆蓋 pred 開關
    if args.enable_pred is not None or os.environ.get("ENABLE_PRED_PATIENT") is not None:
        v = (args.enable_pred if args.enable_pred is not None else os.environ.get("ENABLE_PRED_PATIENT", "1"))
        globals()["ENABLE_PRED_PATIENT"] = True if str(v).strip().lower() in ("1", "true", "yes") else False

    # 覆蓋 pred json
    pj = args.pred_json or os.environ.get("PRED_PATIENT_INPUT_JSON", "")
    if pj:
        try:
            obj = json.loads(pj)
            if isinstance(obj, dict):
                globals()["PRED_PATIENT_INPUT"] = obj
        except Exception:
            pass

    df_raw0 = load_or_demo(input_path)
   

    if df_raw0.columns[0] != "ID":
        df_raw0 = df_raw0.rename(columns={df_raw0.columns[0]: "ID"})

    df_raw0["ID"] = df_raw0["ID"].astype(str)

    # find Profile column (case-insensitive)
    profile_candidates = [c for c in df_raw0.columns if c.lower() == "profile"]
    if len(profile_candidates) != 1:
        raise RuntimeError("Expect a Profile column named 'Profile' (case-insensitive).")
    profile_col = profile_candidates[0]

    # create pred patient row
    df_pred_patient = None
    df_raw_train = df_raw0.copy()

    # numeric item columns
    item_cols = [c for c in df_raw0.columns if c not in ("ID", profile_col)]
    if ENABLE_PRED_PATIENT:
        if PRED_PATIENT_INPUT is not None:
            df_pred_patient = make_dummy_patient_from_input(df_raw0, item_cols, PRED_PATIENT_INPUT,
                                                           PRED_PATIENT_ID, profile_col)
            print(f"[PRED] dummy patient created from PRED_PATIENT_INPUT: {PRED_PATIENT_ID} (excluded from training)")
        else:
            if len(df_raw0) >= 4:
                orig_id = str(df_raw0["ID"].iloc[0])
                df_pred_patient = df_raw0.iloc[[0]].copy()
                df_pred_patient.loc[:, "ID"] = PRED_PATIENT_ID
                df_pred_patient.loc[:, profile_col] = np.nan
                df_raw_train = df_raw0.iloc[1:].copy()
                print(f"[PRED] using first row as prediction-only patient: {orig_id} -> {PRED_PATIENT_ID} (removed from training)")
            else:
                df_pred_patient = make_dummy_patient_row(df_raw0, item_cols, PRED_PATIENT_ID, profile_col, PRED_NOISE_SD_FRAC)
                print(f"[PRED] dataset too small to steal first row; dummy created from column means: {PRED_PATIENT_ID}")

    df_raw = df_raw_train.copy()
    df_raw = df_raw.set_index("ID", drop=True)

    # TRUE_LABEL vector
    true_label = pd.to_numeric(df_raw[profile_col], errors="coerce").astype("Int64").to_numpy()
    u_y = sorted(pd.Series(true_label).dropna().astype(int).unique().tolist())
    if len(u_y) < 2:
        raise RuntimeError("TRUE_LABEL(Profile) must have at least 2 unique (non-NA) classes.")
    target_clusters = max(2, len(u_y))

    # df_num
    df_num = df_raw.drop(columns=[profile_col]).copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    # drop zero variance
    sds = df_num.std(axis=0, skipna=True)
    keep = sds[np.isfinite(sds) & (sds > 0)].index.tolist()
    df_num = df_num[keep].copy()

    if df_num.shape[1] < 2 or df_num.shape[0] < 3:
        raise RuntimeError("Need at least 3 rows and 2 numeric items after cleaning.")

    # -------------------------
    # Scale block (once)
    # -------------------------
    sb = {}
    try:
        df_imp = impute_mean_df(df_num)
        alpha_vals = cronbach_alpha(df_imp)
        alpha_tbl = pd.DataFrame([{"Metric": k, "Value": v} for k, v in alpha_vals.items()])
        C = np.corrcoef(df_imp.values.astype(float), rowvar=False)
        eigs = np.linalg.eigvalsh(C)[::-1]
        aac = compute_aac_top3(eigs)
        aac_tbl = pd.DataFrame([aac])
        eig_tbl = pd.DataFrame({"Idx": np.arange(1, min(20, len(eigs)) + 1), "Eigenvalue": eigs[:min(20, len(eigs))]})
        pa_png = os.path.join(FIG_DIR, "PA_scree.png")
        pa_path = os.path.join("figures", save_pa_scree(df_imp, target_clusters, pa_png, n_iter=20, seed=123))
        efa_load, efa_phi = run_efa_optional(df_imp, target_clusters)
        sb = {
            "alpha_tbl": alpha_tbl,
            "pa_path": pa_path,
            "aac_tbl": aac_tbl,
            "eig_tbl": eig_tbl,
            "efa_loadings": efa_load,
            "efa_phi": efa_phi
        }
    except Exception as e:
        print("[WARN] scale block failed:", str(e))
        sb = {}

    scale_block_html = build_scale_block_html(sb)

    # -------------------------
    # RUN modes A/B/C
    # -------------------------
    res: Dict[str, Any] = {}
    for m in ["A", "B", "C"]:
        print(f"\n[RUN] Mode = {m}")
        res[m] = run_one_mode(m, df_num, df_raw, true_label.astype(float), target_clusters, u_y,
                              min_wcd=1, top_k=20, verbose=True)

    # -------------------------
    # BASELINES
    # -------------------------
    efa_loadings_df = sb.get("efa_loadings") if sb else None
    factor_assign = build_factor_assignment_from_efa(efa_loadings_df, items=df_num.columns.tolist(), K=target_clusters, loading_floor=0.0)
    factor_sums_df = compute_factor_sums(df_num, factor_assign["item_factor"], K=factor_assign["K"], scale_items=True)

    print("\n[RUN] BASELINE = ARGMAX")
    res["ARGMAX"] = run_baseline_mode("ARGMAX", df_num, df_raw, true_label.astype(float), target_clusters, factor_assign, factor_sums_df)
    print("\n[RUN] BASELINE = KMEANS")
    res[f"KMEANS{factor_assign['K']}"] = run_baseline_mode("KMEANS", df_num, df_raw, true_label.astype(float), target_clusters, factor_assign, factor_sums_df)
    print("\n[RUN] BASELINE = AUGMAX")
    res["AUGMAX"] = run_baseline_mode("AUGMAX", df_num, df_raw, true_label.astype(float), target_clusters, factor_assign, factor_sums_df)
    print("\n[RUN] BASELINE = KMEANS_AUGMAX")
    res[f"KMEANS_AUGMAX{factor_assign['K']}"] = run_baseline_mode("KMEANS_AUGMAX", df_num, df_raw, true_label.astype(float), target_clusters, factor_assign, factor_sums_df)

    # -------------------------
    # Write report
    # -------------------------
    rep = write_report_html(scale_block_html, res, OUT_DIR, df_num, target_clusters, df_pred_patient)
    print(f"\n[OK] Best mode = {rep['best_mode']}")
    print("[OK] report.html:", os.path.abspath(rep["latest"]))
    print("[OK] timestamp report:", os.path.abspath(rep["timestamp"]))
    print("\n[DONE] Outputs are in:", os.path.abspath(OUT_DIR))


# webapp.py
import os
import re
import sys
import json
import uuid
import shutil
import subprocess
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, request, send_from_directory, jsonify, abort, redirect

# pandas 用來把 "80,70,60" 對應到欄位名稱 -> pred_json
try:
    import pandas as pd
except Exception:
    pd = None

BASE_DIR = Path(__file__).resolve().parent
MAIN_PY = BASE_DIR / "main.py"

REPORTS_ROOT = BASE_DIR / "web_reports"
UPLOADS_ROOT = BASE_DIR / "web_uploads"
STATIC_DIR  = BASE_DIR / "static"

REPORTS_ROOT.mkdir(exist_ok=True)
UPLOADS_ROOT.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


def safe_name(name: str) -> str:
    name = name or "upload.csv"
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] if len(name) > 180 else name


def make_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def valid_job_id(job_id: str) -> bool:
    return re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", job_id or "") is not None


def find_demo_file() -> Path | None:
    # 你可以把 demo 放在 static 裡，這裡會自動找
    candidates = [
        STATIC_DIR / "demo.csv",
        STATIC_DIR / "demo.tsv",
        STATIC_DIR / "sample.csv",
        STATIC_DIR / "sample.tsv",
        STATIC_DIR / "data_demo.csv",
        STATIC_DIR / "zoodigitaldata.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_pred_seq_to_json(pred_seq: str, input_path: str) -> dict:
    """
    pred_seq: "80,70,60,,72,55"  (空白/NA 會跳過)
    會依照 input 檔案的 item 欄位順序，轉成 {"Math":80, "Physics":70, ...}
    """
    if not pred_seq.strip():
        return {}

    if pd is None:
        raise RuntimeError("pandas is not installed; cannot map pred_seq to column names.")

    ext = Path(input_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    else:
        # csv/tsv/txt：自動猜分隔
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(input_path, sep=sep, engine="python")

    if df.shape[0] < 1:
        raise RuntimeError("Input file has no rows.")

    cols = list(df.columns)

    # 排除 ID / Profile(=TRUE_LABEL)
    def is_id(c): return str(c).strip().lower() == "id"
    def is_profile(c): return str(c).strip().lower() == "profile"

    item_cols = [c for c in cols if (not is_id(c)) and (not is_profile(c))]

    # 拆 pred_seq
    raw = pred_seq.strip()
    parts = [x.strip() for x in raw.split(",")]

    pred = {}
    for i, col in enumerate(item_cols):
        if i >= len(parts):
            break
        v = parts[i]
        if v == "" or v.lower() in ("na", "nan", "null", "none"):
            continue
        try:
            pred[col] = float(v)
        except Exception:
            # 如果有人輸入了非數字，就略過
            continue

    return pred


@app.get("/ping")
def ping():
    return jsonify({"ok": True, "time": datetime.now().isoformat()})


@app.get("/")
def home():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.get("/reports/<job_id>/")
def reports_job_root(job_id: str):
    if not valid_job_id(job_id):
        abort(404)
    return redirect(f"/reports/{job_id}/report.html", code=302)


@app.get("/reports/<job_id>/<path:filename>")
def serve_report_files(job_id: str, filename: str):
    if not valid_job_id(job_id):
        abort(404)
    job_dir = REPORTS_ROOT / job_id
    if not job_dir.exists():
        abort(404)
    return send_from_directory(str(job_dir), filename)


@app.post("/run")
def run_pipeline():
    """
    Supports:
      - upload: multipart form with file field "file"
      - demo: form field use_demo=1
    Dummy patient input:
      - pred_seq: "80,70,60,,72,55"  (推薦給一般使用者)
      - pred_json: JSON object (進階)
    """
    use_demo = (request.form.get("use_demo", "0") or "").strip().lower() in ("1", "true", "yes")
    enable_pred = (request.form.get("enable_pred", "1") or "").strip().lower() in ("1", "true", "yes")

    pred_seq = (request.form.get("pred_seq", "") or "").strip()
    pred_json = (request.form.get("pred_json", "") or "").strip()

    # 先驗證 pred_json（如果有）
    pred_obj = None
    if pred_json:
        try:
            pred_obj = json.loads(pred_json)
            if not isinstance(pred_obj, dict):
                return jsonify({"ok": False, "error": "pred_json must be a JSON object/dict."}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": f"Invalid pred_json: {e}"}), 400

    f = request.files.get("file")

    # 建 job
    job_id = make_job_id()
    job_dir = REPORTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = None

    # 1) upload 模式
    if f and f.filename:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"ok": False, "error": f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXT)}"}), 400

        upload_name = safe_name(f.filename)
        upload_path = UPLOADS_ROOT / f"{job_id}_{upload_name}"
        f.save(str(upload_path))
        input_path = str(upload_path)

    # 2) demo 模式
    if input_path is None:
        if not use_demo:
            return jsonify({"ok": False, "error": "No file uploaded. Use Run Demo or set use_demo=1."}), 400

        demo_file = find_demo_file()
        if demo_file:
            demo_copy = UPLOADS_ROOT / f"{job_id}_{demo_file.name}"
            shutil.copyfile(str(demo_file), str(demo_copy))
            input_path = str(demo_copy)
        else:
            # 如果你 main.py 本身支援「沒帶 --input 就跑 demo」，可以留空
            input_path = None

    if not MAIN_PY.exists():
        return jsonify({"ok": False, "error": f"main.py not found at: {MAIN_PY}"}), 500

    # pred_seq -> pred_json（如果使用者沒給 JSON）
    if (pred_obj is None) and pred_seq:
        if input_path is None:
            return jsonify({"ok": False, "error": "Demo pred_seq needs a demo file in ./static (e.g., demo.csv) to know column order."}), 400
        try:
            pred_obj = parse_pred_seq_to_json(pred_seq, input_path)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Cannot parse pred_seq: {e}"}), 400

    # 組 cmd：盡量走 CLI args（比 env 穩）
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--outdir", str(job_dir),
        "--enable_pred", ("1" if enable_pred else "0"),
    ]
    if input_path:
        cmd += ["--input", input_path]
    if pred_obj:
        cmd += ["--pred_json", json.dumps(pred_obj, ensure_ascii=False)]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )

    # log
    log_path = job_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as w:
        w.write("=== CMD ===\n")
        w.write(" ".join(cmd) + "\n\n")
        w.write("=== STDOUT ===\n")
        w.write(proc.stdout or "")
        w.write("\n\n=== STDERR ===\n")
        w.write(proc.stderr or "")

    if proc.returncode != 0:
        return jsonify({
            "ok": False,
            "job_id": job_id,
            "error": "Pipeline failed (non-zero exit code).",
            "report_url": f"/reports/{job_id}/report.html",
            "log_url": f"/reports/{job_id}/run.log",
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }), 500

    # 確認 report.html
    report_path = job_dir / "report.html"
    if not report_path.exists():
        cand = sorted(job_dir.glob("report_*.html"))
        if cand:
            shutil.copyfile(str(cand[-1]), str(report_path))

    if not report_path.exists():
        return jsonify({
            "ok": False,
            "job_id": job_id,
            "error": "Pipeline finished but report.html was not found in OUT_DIR.",
            "log_url": f"/reports/{job_id}/run.log",
        }), 500

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "report_url": f"/reports/{job_id}/report.html",
        "log_url": f"/reports/{job_id}/run.log",
        "stdout_tail": (proc.stdout or "")[-4000:],
    })


def run_dev_server():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    auto_open = os.environ.get("AUTO_OPEN", "1").strip().lower() in ("1", "true", "yes")
    if auto_open:
        url = f"http://{host}:{port}/"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_dev_server()
