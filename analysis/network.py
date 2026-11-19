import networkx as nx
import pandas as pd


CARRIERS = {
    "Coal": ["原煤", "洗精煤", "煤矸石", "型煤", "焦炭"],
    "Oil": ["原油", "汽油", "柴油", "燃料油", "煤油", "石脑油", "润滑油"],
    "Gas": ["天然气", "液化天然气", "液化石油气", "高炉煤气", "焦炉煤气", "转炉煤气"],
    "Electricity": ["电力"],
    "Heat": ["热力"],
    "Refinery": ["炼厂干气", "石蜡", "溶剂油", "石油沥青", "石油焦"],
    "Others": ["其他能源", "其他石油制品", "其他洗煤", "其他煤气", "其他焦化产品"],
}
CARRIER_CATEGORY = {node: category for category, nodes in CARRIERS.items() for node in nodes}
FINAL_USE = {
    "工业终端消费量",
    "交通运输、仓储和邮政业终端消费量",
    "建筑业终端消费量",
    "批发、零售业和住宿、餐饮业终端消费量",
    "生活消费-城镇生活消费",
    "生活消费-乡村生活消费",
    "农、林、牧、渔_、水利业终端消费量",
    "其他终端消费量",
}
NATIONAL_EXCLUSIONS = {"外省_区、市调入量", "本省_区、市调出量_-", "损失量"}
CORE = ["Coal", "Oil", "Gas", "Refinery"]
SCOPES = {
    "core_fossil": CORE,
    "core_plus_heat": [*CORE, "Heat"],
    "extended_legacy": [*CORE, "Heat", "Others"],
}
CASES = {"浙江": (2019, 2020), "上海": (2004, 2005), "海南": (2010, 2011)}


def build_graphs(balance, nodes):
    graphs = {}
    for key, rows in balance.groupby(["province", "year"], sort=False):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for row in rows.itertuples(index=False):
            value = row.signed_10k_tce
            if pd.isna(value) or value == 0:
                continue
            source, target = (row.account, row.carrier) if value > 0 else (row.carrier, row.account)
            graph.add_edge(source, target, weight=abs(value))
        graphs[key] = graph
    return graphs


def add_flow(graph, source, target, weight):
    edge = graph.get_edge_data(source, target, {"weight": 0.0})
    graph.add_edge(source, target, weight=edge["weight"] + weight)


def dominant_category(graph):
    aggregated = nx.DiGraph()
    for source, target, weight in graph.edges(data="weight"):
        add_flow(aggregated, CARRIER_CATEGORY.get(source, source),
                 CARRIER_CATEGORY.get(target, target), abs(weight))
    best = None
    best_pair = (0.0, 0.0)
    for category in ("Coal", "Gas", "Oil"):
        outgoing = [(weight, target) for _, target, weight in aggregated.out_edges([category], data="weight")
                    if target in FINAL_USE]
        if outgoing:
            weight, target = max(outgoing, key=lambda pair: pair[0])
            candidate = weight, weight / aggregated.in_degree(target, weight="weight")
            if candidate > best_pair:
                best, best_pair = category, candidate
    return best, best_pair[1]


def subnetwork(graph, category):
    seeds = set(CARRIERS[category]).intersection(graph)
    nodes = seeds.copy()
    for seed in seeds:
        nodes.update(graph.predecessors(seed))
        nodes.update(graph.successors(seed))
    return graph.subgraph(nodes)


def graph_metrics(graph):
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    return {
        "nodes": nodes,
        "edges": edges,
        "mean_degree": 2 * edges / nodes if nodes else 0.0,
        "mean_weighted_degree": sum(degree for _, degree in graph.degree(weight="weight")) / nodes if nodes else 0.0,
    }


def national_diagnostics(graphs):
    national = {}
    for (_, year), graph in graphs.items():
        merged = national.setdefault(year, nx.DiGraph())
        merged.add_nodes_from(graph)
        for source, target, weight in graph.edges(data="weight"):
            if source not in NATIONAL_EXCLUSIONS and target not in NATIONAL_EXCLUSIONS:
                add_flow(merged, source, target, weight)

    rows = []
    for year, graph in sorted(national.items()):
        metrics = {category: graph_metrics(subnetwork(graph, category)) for category in CARRIERS}
        for scope, categories in SCOPES.items():
            parts = [metrics[category] for category in categories]
            nodes = sum(part["nodes"] for part in parts)
            edges = sum(part["edges"] for part in parts)
            weighted_degree = sum(part["nodes"] * part["mean_weighted_degree"] for part in parts)
            rows.append((scope, year, nodes, edges, 2 * edges / nodes if nodes else 0.0,
                         weighted_degree / nodes if nodes else 0.0))
    return pd.DataFrame(rows, columns=[
        "scope", "year", "category_summed_nodes", "category_summed_edges",
        "node_weighted_mean_degree", "node_weighted_mean_weighted_degree",
    ])


def national_period_summary(diagnostics):
    metrics = ["category_summed_edges", "node_weighted_mean_degree", "node_weighted_mean_weighted_degree"]
    early = diagnostics.loc[diagnostics.year.between(2001, 2009)].groupby("scope")[metrics].mean().stack()
    late = diagnostics.loc[diagnostics.year.between(2010, 2020)].groupby("scope")[metrics].mean().stack()
    return pd.DataFrame({
        "mean_2001_2009": early, "mean_2010_2020": late,
        "absolute_change": late - early, "percent_change": (late / early.where(early.ne(0)) - 1) * 100,
    }).rename_axis(["scope", "metric"]).reset_index()


def network_cases(graphs, panel):
    stats, edges, overlap = [], [], []
    categories = panel.set_index(["province", "year"])["dominant_category"]
    for province, years in CASES.items():
        if not all((province, year) in graphs and (province, year) in categories.index for year in years):
            continue
        edge_pairs = []
        for year in years:
            graph = graphs[province, year]
            category = categories.loc[province, year]
            part = subnetwork(graph, category)
            stats.append({"province": province, "year": year, "category": category,
                          **graph_metrics(part), "drawn_nodes": len(part) - len(list(nx.isolates(part))),
                          "full_network_nodes": len(graph), "full_network_edges": graph.number_of_edges()})
            edge_pairs.append(set(part.edges))
            edges.extend((province, year, source, target, weight)
                         for source, target, weight in part.edges(data="weight"))
        overlap.append((province, *years, len(edge_pairs[0]), len(edge_pairs[1]),
                        len(edge_pairs[0] & edge_pairs[1])))
    return (
        pd.DataFrame(stats, columns=["province", "year", "category", "nodes", "edges", "mean_degree",
                                     "mean_weighted_degree", "drawn_nodes", "full_network_nodes", "full_network_edges"]),
        pd.DataFrame(edges, columns=["province", "year", "source", "target", "flow_10k_tce"]),
        pd.DataFrame(overlap, columns=["province", "earlier_year", "later_year", "earlier_edges", "later_edges",
                                      "identical_directed_edges"]),
    )
