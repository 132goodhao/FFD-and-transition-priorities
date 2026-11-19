import argparse
from pathlib import Path

import pandas as pd

from .accounts import electricity_balance, industrial_context, thermal_power
from .ffd import ffd_panel, persistence_sensitivity
from .network import build_graphs, national_diagnostics, national_period_summary, network_cases
from .roles import provincial_roles, supply_flows
from .transitions import annual_transitions


def save_tables(output, tables):
    output.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        data.to_csv(output / f"{name}.csv", index=False)


def read_panel(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name="S1_FFD_panel", header=4).rename(columns={
            "Province": "province_en", "Province_CN": "province", "Year": "year", "FFD": "ffd",
            "Dominant_fossil_category": "dominant_category", "Dominant_flow_share": "flow_share",
            "Dominant_category_persistence": "persistence", "Mean_unweighted_degree": "mean_degree",
        })
    return pd.read_csv(path)


def read_graphs(balance_path, node_path):
    balance = pd.read_csv(balance_path)
    nodes = pd.read_csv(node_path).node
    if balance[["province", "year", "carrier", "account"]].isna().any().any() or nodes.isna().any():
        raise ValueError("Province, year, and node labels are required")
    unknown = (set(balance.carrier) | set(balance.account)) - set(nodes)
    if unknown:
        raise ValueError(f"Balance labels missing from node list: {sorted(unknown)}")
    return build_graphs(balance, nodes)


def transition_tables(panel):
    events, counts, shares, thresholds = annual_transitions(panel)
    overview = panel.groupby("year").ffd.agg(["mean", "median", "min", "max"]).reset_index()
    return {
        "ffd_annual_summary": overview,
        "ffd_state_thresholds": thresholds.rename("ffd").rename_axis("quantile").reset_index(),
        "markov_events": events,
        "markov_matrices": counts,
        "markov_state_shares": shares,
    }


def network_command(args):
    graphs = read_graphs(args.balance, args.nodes)
    panel, weights = ffd_panel(graphs)
    stats, edges, overlap = network_cases(graphs, panel)
    national = national_diagnostics(graphs)
    save_tables(args.output, {
        "ffd_panel": panel,
        "ffd_weights": weights.rename("weight").rename_axis("component").reset_index(),
        "ffd_persistence_sensitivity": persistence_sensitivity(panel),
        "national_subnetworks": national,
        "national_subnetwork_period_summary": national_period_summary(national),
        "provincial_subnetwork_stats": stats,
        "provincial_subnetwork_edges": edges,
        "provincial_subnetwork_overlap": overlap,
        **transition_tables(panel),
    })


def roles_command(args):
    if args.flows:
        flows = pd.read_csv(args.flows)
    else:
        if args.nodes is None:
            raise ValueError("--nodes is required with --balance")
        flows = supply_flows(read_graphs(args.balance, args.nodes))
    panel = read_panel(args.panel)
    indices, carriers, channels, weights = provincial_roles(flows, panel)
    save_tables(args.output, {
        "provincial_roles": indices,
        "sri_carrier_contributions": carriers,
        "sri_channel_contributions": channels,
        "eri_sri_alternative_weights": weights,
    })


def cases_command(args):
    cells = pd.read_csv(args.cells, low_memory=False)
    if args.factors.suffix.lower() == ".xlsx":
        factors = pd.read_excel(args.factors, sheet_name="S4_factors", header=4).rename(columns={
            "Raw_carrier": "carrier", "Original_unit": "unit", "Factor_to_10k_tce": "factor_to_10k_tce",
        })
    else:
        factors = pd.read_csv(args.factors)
    thermal, thermal_changes = thermal_power(cells, factors[["carrier", "unit", "factor_to_10k_tce"]])
    electricity, electricity_changes = electricity_balance(cells)
    save_tables(args.output, {
        "thermal_power_10k_tce": thermal,
        "thermal_power_changes_10k_tce": thermal_changes,
        "electricity_100m_kwh": electricity,
        "electricity_changes_100m_kwh": electricity_changes,
        "industrial_context": industrial_context(pd.read_csv(args.series, low_memory=False)),
    })


def supplement_command(args):
    panel = read_panel(args.workbook)
    save_tables(args.output, {
        **transition_tables(panel),
        "ffd_persistence_sensitivity": persistence_sensitivity(panel),
    })


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    network = commands.add_parser("network")
    network.add_argument("--balance", type=Path, required=True)
    network.add_argument("--nodes", type=Path, required=True)
    network.set_defaults(run=network_command)

    roles = commands.add_parser("roles")
    sources = roles.add_mutually_exclusive_group(required=True)
    sources.add_argument("--flows", type=Path)
    sources.add_argument("--balance", type=Path)
    roles.add_argument("--nodes", type=Path)
    roles.add_argument("--panel", type=Path, required=True)
    roles.set_defaults(run=roles_command)

    cases = commands.add_parser("cases")
    cases.add_argument("--cells", type=Path, required=True)
    cases.add_argument("--factors", type=Path, required=True)
    cases.add_argument("--series", type=Path, required=True)
    cases.set_defaults(run=cases_command)

    supplement = commands.add_parser("supplement")
    supplement.add_argument("--workbook", type=Path, required=True)
    supplement.set_defaults(run=supplement_command)

    for command in (network, roles, cases, supplement):
        command.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
