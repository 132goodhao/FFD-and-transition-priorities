import numpy as np
import pandas as pd

from .network import dominant_category, graph_metrics, subnetwork
from .transitions import assign_states


COMPONENTS = ["flow_share", "persistence", "mean_degree"]
MISSING_YEARS = [("宁夏", 2001), ("宁夏", 2002), ("海南", 2002)]


def entropy_topsis(panel):
    values = panel[COMPONENTS].astype(float)
    span = values.max() - values.min()
    normalized = (values - values.min()).div(span.where(span > 0)).fillna(0)
    if len(normalized) < 2:
        raise ValueError("TOPSIS requires at least two observations")
    shares = normalized / (normalized.sum() + 1e-12)
    entropy = -(shares * np.log(shares + 1e-12)).sum() / np.log(len(normalized))
    information = 1 - entropy
    weights = information / information.sum()
    weighted = normalized * weights
    positive = np.sqrt(((weighted - weighted.max()) ** 2).sum(axis=1))
    negative = np.sqrt(((weighted - weighted.min()) ** 2).sum(axis=1))
    return negative / (positive + negative + 1e-12), weights


def ffd_panel(graphs):
    rows = []
    for (province, year), graph in graphs.items():
        category, share = dominant_category(graph)
        if category is not None:
            rows.append((province, year, category, share, graph_metrics(subnetwork(graph, category))["mean_degree"]))
    panel = pd.DataFrame(rows, columns=["province", "year", "dominant_category", "flow_share", "mean_degree"])
    panel["persistence"] = (
        panel.groupby(["province", "dominant_category"])["year"].transform("size")
        / panel.groupby("province")["year"].transform("size")
    )
    panel["imputed"] = False
    for province, year in MISSING_YEARS:
        region = panel.loc[panel.province.eq(province)]
        if year not in region.year.values:
            substitute = region.loc[region.year.eq(2003)].assign(year=year, imputed=True)
            panel = pd.concat([panel, substitute], ignore_index=True)
    panel = panel.sort_values(["province", "year"]).reset_index(drop=True)
    panel["ffd"], weights = entropy_topsis(panel)
    return panel, weights


def persistence_sensitivity(panel):
    revised = panel.copy()
    hainan = revised.province.eq("海南")
    revised.loc[hainan, "persistence"] = revised.loc[hainan, "dominant_category"].map(
        revised.loc[hainan, "dominant_category"].value_counts(normalize=True)
    )
    revised["ffd_recalculated"], _ = entropy_topsis(revised)
    revised["delta_ffd"] = revised.ffd_recalculated - revised.ffd
    quantiles = [0.33, 0.67]
    thresholds = revised.ffd.quantile(quantiles)
    revised["original_state"] = assign_states(revised.ffd, thresholds)
    revised["recalculated_state_fixed_thresholds"] = assign_states(revised.ffd_recalculated, thresholds)
    revised["recalculated_state_new_thresholds"] = assign_states(
        revised.ffd_recalculated, revised.ffd_recalculated.quantile(quantiles)
    )
    return revised
