import numpy as np
import pandas as pd


PROVINCES = ["河北", "山东"]
PROCESS_GASES = {"焦炉煤气", "高炉煤气", "转炉煤气"}
COAL_SOLIDS = {"原煤", "洗精煤", "其他洗煤", "煤矸石", "型煤", "焦炭", "其他焦化产品"}
NATURAL_GAS = {"天然气", "液化天然气"}
OIL = {
    "原油", "汽油", "煤油", "柴油", "燃料油", "石脑油", "润滑油", "石蜡", "溶剂油",
    "石油沥青", "石油焦", "液化石油气", "炼厂干气", "其他石油制品",
}
POWER_GROUPS = {
    **dict.fromkeys(PROCESS_GASES, "process_gases"),
    **dict.fromkeys(COAL_SOLIDS, "other_coal_related"),
    **dict.fromkeys(NATURAL_GAS, "natural_gas"),
    **dict.fromkeys(OIL, "oil_products"),
    "热力": "heat_input",
}
POWER_INPUTS = ["process_gases", "other_coal_related", "natural_gas", "oil_products", "heat_input", "other_energy"]
ELECTRICITY = {
    "一次能源生产量": "nonthermal_generation",
    "火力发电": "thermal_generation",
    "终端消费量": "final_use",
}
PHYSICAL_ITEMS = {
    **ELECTRICITY, "外省(区、市)调入量": "inflow", "本省(区、市)调出量(-)": "outflow",
    "消费量合计": "total_use",
}
STANDARD_ITEMS = {**ELECTRICITY, "调入量": "inflow", "调出量": "outflow", "损失量": "loss"}
FLOW_METRICS = ["nonthermal_generation", "thermal_generation", "inflow", "outflow", "total_use", "final_use"]
INDICATORS = {
    "发电量(亿千瓦时)": "total_generation_100m_kwh",
    "火力发电量(亿千瓦时)": "thermal_generation_100m_kwh",
    "水力发电量(亿千瓦时)": "hydro_generation_100m_kwh",
    "电力消费量(亿千瓦时)": "electricity_consumption_100m_kwh",
    "能源-主要能源产品产量-焦炭产量(万吨)": "coke_output_10k_tonnes",
    "生铁产量(万吨)": "pig_iron_output_10k_tonnes",
    "粗钢产量(万吨)": "crude_steel_output_10k_tonnes",
}


def thermal_power(cells, factors):
    ledger = cells.loc[
        cells.province.isin(PROVINCES) & cells.year.isin([2021, 2022])
        & cells.basis.eq("physical") & cells.item.eq("火力发电") & ~cells.is_aggregate_column
    ].copy()
    ledger["carrier"] = ledger.carrier.replace({"炼厂千气": "炼厂干气"})
    ledger = ledger.merge(factors, on=["carrier", "unit"], how="left", validate="many_to_one")
    ledger["value"] = pd.to_numeric(ledger.value, errors="raise")
    if ledger.loc[ledger.value.notna(), "factor_to_10k_tce"].isna().any():
        raise ValueError("Missing conversion factor for a reported thermal-power observation")
    ledger["signed_10k_tce"] = ledger.value * ledger.factor_to_10k_tce
    ledger["carrier"] = ledger.carrier.replace({"煤制品": "型煤"})
    ledger["group"] = ledger.carrier.map(POWER_GROUPS).fillna("other_energy")
    keys = ["province", "year"]
    inputs = ledger.loc[ledger.signed_10k_tce.lt(0)].copy()
    inputs["amount_10k_tce"] = -inputs.signed_10k_tce
    by_group = inputs.groupby([*keys, "group"]).amount_10k_tce.sum().unstack(fill_value=0)
    by_group = by_group.reindex(columns=POWER_INPUTS, fill_value=0)
    outputs = ledger.loc[ledger.signed_10k_tce.gt(0) & ledger.carrier.eq("电力")].groupby(keys).signed_10k_tce.sum()
    by_group["other_recorded_inputs"] = by_group[POWER_INPUTS[1:]].sum(axis=1)
    by_group["total_inputs"] = by_group[POWER_INPUTS].sum(axis=1)
    by_group["thermal_output"] = outputs
    missing = by_group.reindex(pd.MultiIndex.from_product([PROVINCES, [2021, 2022]], names=keys)).isna().any().any()
    if missing:
        raise ValueError("Thermal-power accounts require both case provinces in 2021 and 2022")
    changes = by_group.xs(2022, level="year") - by_group.xs(2021, level="year")
    return by_group.reset_index(), changes.add_prefix("delta_").reset_index()


def electricity_rows(cells, years, basis, items):
    selected = cells.loc[
        cells.province.isin(PROVINCES) & cells.year.isin(years)
        & cells.basis.eq(basis) & cells.carrier.eq("电力") & cells.item.isin(items)
    ].copy()
    selected["metric"] = selected.item.map(items)
    selected["value"] = pd.to_numeric(selected.value, errors="raise")
    return selected.pivot(index=["province", "year"], columns="metric", values="value").reindex(columns=items.values())


def electricity_balance(cells):
    physical = electricity_rows(cells, [2021, 2022], "physical", PHYSICAL_ITEMS)
    physical[["inflow", "outflow"]] = physical[["inflow", "outflow"]].abs().fillna(0)
    standard = electricity_rows(cells, [2023], "standard_coal_equivalent", STANDARD_ITEMS) / 1.229
    standard[["inflow", "outflow"]] = standard[["inflow", "outflow"]].abs()
    standard["total_use"] = standard.final_use + standard.loss.fillna(0)
    balance = pd.concat([physical[FLOW_METRICS], standard[FLOW_METRICS]])
    expected = pd.MultiIndex.from_product([PROVINCES, [2021, 2022, 2023]], names=["province", "year"])
    balance = balance.reindex(expected)
    if balance.isna().any().any():
        raise ValueError("Incomplete electricity accounts for the two provinces")
    balance["loss_and_internal_use"] = balance.total_use - balance.final_use
    balance["balance_residual"] = (
        balance.nonthermal_generation + balance.thermal_generation + balance.inflow - balance.outflow - balance.total_use
    )
    changes = balance.groupby(level="province").diff().dropna()
    changes["export_supply_effect"] = -changes.outflow
    return balance.reset_index(), changes.add_prefix("delta_").reset_index()


def industrial_context(series):
    selected = series.loc[
        series.province.isin(PROVINCES) & series.year.between(2018, 2024) & series.indicator.isin(INDICATORS)
    ].copy()
    selected["metric"] = selected.indicator.map(INDICATORS)
    selected["value"] = pd.to_numeric(selected.value, errors="raise")
    panel = selected.pivot(index=["province", "year"], columns="metric", values="value")
    expected = pd.MultiIndex.from_product([PROVINCES, range(2018, 2025)], names=["province", "year"])
    panel = panel.reindex(index=expected, columns=INDICATORS.values())
    if panel.isna().any().any():
        raise ValueError("Incomplete case-study series for 2018–2024")
    panel["nonthermal_generation_100m_kwh"] = panel.total_generation_100m_kwh - panel.thermal_generation_100m_kwh
    panel["other_nonthermal_generation_100m_kwh"] = panel.nonthermal_generation_100m_kwh - panel.hydro_generation_100m_kwh
    panel["thermal_generation_share"] = panel.thermal_generation_100m_kwh / panel.total_generation_100m_kwh
    panel["local_generation_gap_100m_kwh"] = panel.total_generation_100m_kwh - panel.electricity_consumption_100m_kwh
    return panel.reset_index()
