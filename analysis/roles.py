import numpy as np
import pandas as pd


CARRIERS = ["原油", "天然气", "原煤", "电力", "液化天然气", "汽油", "柴油", "煤油", "燃料油"]
CHANNELS = ["exports", "outflows", "production"]
CAPS = pd.Series({"exports": 1000.0, "outflows": 1000.0, "production": 5000.0})
ERI_COMPONENTS = ["import_dependency", "net_import_ratio", "self_sufficiency_deficit"]
ROLES = {
    (True, True): "External input and supply coordination",
    (True, False): "External input",
    (False, True): "Supply coordination",
    (False, False): "More locally oriented",
}


def supply_flows(graphs):
    rows = []
    for (province, year), graph in graphs.items():
        if year != 2020:
            continue
        for carrier in CARRIERS:
            edges = [
                ("进口量", carrier),
                ("外省_区、市调入量", carrier),
                (carrier, "出口量_-"),
                (carrier, "本省_区、市调出量_-"),
                ("一次能源生产量", carrier),
            ]
            amounts = [abs(graph.get_edge_data(*edge, default={"weight": 0.0})["weight"]) for edge in edges]
            rows.append((province, carrier, *amounts))
    return pd.DataFrame(rows, columns=["province", "carrier", "imports", "inflows", "exports", "outflows", "production"])


def entropy_weights(values):
    span = values.max() - values.min()
    normalized = (values - values.min()).div(span.where(span > 1e-12)).fillna(0.0) + 0.0001
    shares = normalized.div(normalized.sum())
    entropy = -(shares * np.log(shares)).sum() / np.log(len(values))
    information = 1 - entropy
    return information / information.sum() if information.sum() > 1e-12 else pd.Series(
        1 / len(values.columns), index=values.columns
    )


def scores(flows):
    data = flows.copy()
    province = data.province
    supply = data.imports + data.inflows + data.production
    supply_total = supply.groupby(province).sum().replace(0, np.nan)
    eri = ((data.imports + data.inflows).groupby(province).sum() / supply_total).clip(0, 1)
    scored = data[CHANNELS].div(CAPS).clip(0, 1)
    scale = data[CHANNELS].clip(lower=0).sum(axis=1)
    scale_total = scale.groupby(province).transform("sum").replace(0, np.nan)
    shares = scale / scale_total
    contributions = scored.mul(shares / 3, axis=0)
    sri_by_carrier = contributions.sum(axis=1, min_count=1)
    sri = sri_by_carrier.groupby(province).sum(min_count=1)
    alt_sri_weights = entropy_weights(data[CHANNELS])
    alt_sri = scored.mul(alt_sri_weights).sum(axis=1).mul(shares).groupby(province).sum(min_count=1)

    trade = (data.imports + data.exports).replace(0, np.nan)
    components = pd.DataFrame({
        "import_dependency": (data.imports / supply.replace(0, np.nan)).fillna(0),
        "net_import_ratio": ((data.imports - data.exports) / trade).fillna(0),
        "self_sufficiency_deficit": 1 - (data.production / supply.replace(0, np.nan)).fillna(0),
    })
    alt_eri_weights = entropy_weights(components[ERI_COMPONENTS])
    components["net_import_ratio"] = components.net_import_ratio.clip(lower=0)
    carrier_eri = components.mul(alt_eri_weights).sum(axis=1).clip(upper=1)
    alt_eri = carrier_eri.mul(supply).groupby(province).sum() / supply_total
    indices = pd.DataFrame({"eri": eri, "sri": sri, "eri_entropy": alt_eri, "sri_entropy": alt_sri})
    carriers = data[["province", "carrier"]].assign(sri_contribution=sri_by_carrier)
    channels = contributions.groupby(province).sum(min_count=1)
    weights = pd.concat({"ERI": alt_eri_weights, "SRI": alt_sri_weights}).rename("weight").rename_axis(
        ["index_name", "component"]
    )
    return indices, carriers, channels, weights


def provincial_roles(flows, panel):
    indices, carriers, channels, weights = scores(flows)
    ffd = panel.loc[panel.year.eq(2020)].set_index("province").ffd
    indices = indices.join(ffd, how="inner")
    for metric in ("ffd", "eri", "sri", "eri_entropy", "sri_entropy"):
        values = indices[metric]
        indices[f"{metric}_above_mean"] = values > values.mean()
        indices[f"{metric}_rank"] = values.rank(ascending=False, method="min")
    indices["system_role"] = [ROLES[pair] for pair in zip(indices.eri_above_mean, indices.sri_above_mean)]
    return indices.reset_index(), carriers, channels.reset_index(), weights.reset_index()
