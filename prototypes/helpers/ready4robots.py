# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:52:41 2021

@author: plank
"""
import argparse
import gc
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from timeit import default_timer as timer

import igraph as ig
import networkx as nx
import osmnx as ox
import pandas as pd
import psutil
import tqdm
from configobj import ConfigObj

from . import configreader as cfr
from . import extract as ep
from . import osmhelper as oh


def actualsize(input_obj):
    """Determine the size of Complex Objects."""
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size


def add_markers_and_color(cfg, list_of_dicts, counter):
    """Add marker Symbols and colors form the config to the dictionary."""
    colors = cfr.get_colors(cfg)
    markers = cfr.get_markers(cfg)
    extra_markers = cfr.get_extra_markers(cfg)
    markers_len = len(markers)

    # Use config "markers" first then "extra_markers"
    if counter < markers_len:
        city_marker = markers[counter]
    elif (counter - markers_len) < len(extra_markers):
        city_marker = extra_markers[counter - markers_len]
    else:
        city_marker = 1

    city_markers = [city_marker for x in range(0, len(cfr.get_scenarios(cfg)))]

    for color, marker, dictionary in zip(colors, city_markers, list_of_dicts):
        dictionary["color"] = color
        dictionary["marker"] = marker

    return list_of_dicts


def analyse_runtime_optimized(cfg, city, override=False,
                              with_speed=True, simplify=True,
                              which_result=0):
    """Calculates the shortest paths and the population data.
    Write output files instead of returning values.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    override : bool, default=False
        if true does not use old files to speed up calculation.
    with_speed : bool, optional
        if True the edge length will be divided by allowed max speed,
        by default True
    simplify : bool, optional
        osmnx parameter,  
        if True, simplify graph topology with the simplify_graph function,
        by default True
    """
    folder = cfg["Folders"]["out"]
    all_there = False
    if not override:
        all_there = True
        for scenario in cfr.yield_scenarios(cfg):

            file_hist_n = Path(folder, city + "_" +
                               scenario["name"] + "_hist_n.json")
            file_hist_f = Path(folder, city + "_" +
                               scenario["name"] + "_hist_f.json")
            file_hist_d = Path(folder, city + "_" +
                               scenario["name"] + "_hist_d.json")
            file_pop = Path(folder, city + "_" +
                            scenario["name"] + "_pop.json")

            if not (file_hist_n.is_file() and file_hist_f.is_file() and file_hist_d.is_file() and file_pop.is_file()):
                all_there = False

    if all_there and not override:
        print("Skipping analyse_runtime_optimized")
        return

    # load city
    G_nx = oh.get_place(placename=city, simplify=simplify,
                        clean_periphery=True,
                        file_identifier="bicycle-tags", retain_all=True,
                        which_result=which_result)

    secondaryGraphs = []

    # print(cfg, G_nx, city, simplify, override, with_speed)
    values_assign = prep_parallel_assign(
        cfg, G_nx, city, simplify, override, with_speed)
    print("Start Assigning pop")
    with Pool(len(cfr.get_scenarios(cfg))) as pool:
        res = pool.starmap(async_assing_pop, tqdm.tqdm(
            values_assign, total=len(values_assign)))
    for entry in res:
        if entry[0] == "Optimal":
            G_ig = entry[1]
        secondaryGraphs.append(entry[1])
    print("Done Assigning pop")

    values, cores = prep_parallel_runtime_optimized(cfg, G_ig, secondaryGraphs)
    scenario_count = len(cfr.get_scenarios(cfg))  # - 1 # exclude optimal
    hist_n, hist_f, hist_d, pop_n = run_parallel_runtime_optimized(
        values, cores, scenario_count)

    j = 0
    for scenario in cfr.yield_scenarios(cfg):

        file_hist_n = Path(folder, city + "_" +
                           scenario["name"] + "_hist_n.json")
        file_hist_f = Path(folder, city + "_" +
                           scenario["name"] + "_hist_f.json")
        file_hist_d = Path(folder, city + "_" +
                           scenario["name"] + "_hist_d.json")
        file_pop = Path(folder, city + "_" + scenario["name"] + "_pop.json")

        json.dump(hist_n[j], open(file_hist_n, 'w'))
        json.dump(hist_f[j], open(file_hist_f, 'w'))
        json.dump(hist_d[j], open(file_hist_d, 'w'))
        json.dump(pop_n[j], open(file_pop, 'w'))
        j += 1


def append_path_stats(normal, filtered, hist_n, hist_f, hist_d,
                      bin_size_n=1, bin_size_d=1):
    """Append path stats to the existing directories.

    Parameters
    ----------
    normal : numpy Matrix
        Matrix of shortest paths from a Subset of Nodes 
        in the primary Graph.
    filtered : numpy Matrix
        Matrix of shortest paths from a Subset of Nodes
        in the secondary Graph.
    hist_n : dict
        Dictionary of how often a path length 
        was calculated in the primary Graph.
    hist_f : dict
        Dictionary of how often a path length 
        was calculated in the secondary Graph.
    hist_d : dict
        Dictionary of how often a detour was calculated.
    bin_size_n : int, default = 1
        The size of a bucket in the histogram dictionaries. 
    bin_size_d : int, default = 1
        The size of a bucket in the detour dictionary.

    Returns
    -------
    hist_n : dict
        Dictionary of how often a path length 
        was calculated in the primary Graph.
    hist_f : dict
        Dictionary of how often a path length 
        was calculated in the secondary Graph.
    hist_d : dict
        Dictionary of how often a detour was calculated.
    pop_n : dict
        Dictionary of reachable pop count in given timeslots per osmid.
    """

    inf = float("inf")

    for n, f in zip(normal, filtered):
        # if path is unreachable in primary
        if n >= inf:
            hist_n['inf'] = hist_n.get("inf", 0) + 1
            hist_f['inf'] = hist_f.get("inf", 0) + 1
            hist_d['inf'] = hist_d.get("inf", 0) + 1
            continue
        # if node is itself
        if n <= 0.0:
            hist_n['self'] = hist_n.get("self", 0) + 1
            hist_f['self'] = hist_f.get("self", 0) + 1
            hist_d['self'] = hist_d.get("self", 0) + 1
            continue

        e_n = round(n/bin_size_n)
        hist_n[e_n] = hist_n.get(e_n, 0) + 1

        # if path is unreachable in secondary
        if f >= inf:
            hist_f['inf'] += 1
            continue

        e_f = round(f/bin_size_n)
        hist_f[e_f] = hist_f.get(e_f, 0) + 1

        # how much in percent longer
        detour = round(((f / n) * 100/bin_size_d))
        hist_d[detour] = hist_d.get(detour, 0) + 1

    return hist_n, hist_f, hist_d


def async_assing_pop(cfg, H_nx, city,
                     scenario, simplify=True, override=False,
                     with_speed=True):
    """Assign population for each scenario in parallel.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    H_nx : NetworkxGraph
        Unfiltered Graph of the city.
    city : str
        Name of the city for which to do the calculations.
    scenario : dict
        The scenario for which to do the calculations.
    simplify : bool, optional
        osmnx parameter,  
        if True, simplify graph topology with the simplify_graph function,
        by default True
    override : bool, default=False
        if true does not use old files to speed up calculation.
    with_speed : bool, optional
        if True the edge length will be divided by allowed max speed,
        by default True

    Returns
    -------
    name : str
        Name of the scenario
    python igraph
        Graph as igraph.
    """
    H_nx = load_pop_scenario(cfg, H_nx, city,
                             scenario["name"], simplify, override)

    H_ig = nx_to_ig(cfg, H_nx, scenario_name=scenario["name"],
                    with_speed=with_speed)

    return scenario["name"], H_ig


def avg_detour_percentage(df):
    """Calculate the Average detour.

    Parameters
    ----------
    df : DataFrame
        DataFrame of all calculated detours.

    Returns
    -------
    int
        Average detour.
    """

    # print(df)
    df = df.drop(["inf", "self", "controll"], axis=1, errors="ignore")
    df = df.T.reset_index().rename(columns={"index": "length", 0: "ammount"})
    df["multi"] = (df["length"] - 100) * df["ammount"]
    sums = df.sum()

    avg = round((sums["multi"] / (sums["ammount"])) / 100, 4)

    return avg


def calc_scenarios_runtime_optimized(G, secondaryGraphs, start, stop, traveltimes, weights="length"):
    """Calculate the paths and pathstats for the Graph and all scenarios.

    Parameters
    ----------
    G : iGraph
        the primary Graph.
    secondaryGraphs : list
        List of secondary graphs.
    start : int
        Start id of this chunk.
        This is the first id for which 
        to calculate all shortest paths to all other nodes.
    stop : int
        Last id of this chunk.
    traveltimes : list
        List of maximum traveltimes for reachability calculations.
    weights : str, optional
        identifier for the weights to use in the shortest path calculations,
        by default "length"

    Returns
    -------
    hist_n_merged : list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated shoretest path length in the primary graph.
        Contains only entries from id start to id stop.
    hist_f_merged : list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated shoretest path length in the secondary graphs.   
        Contains only entries from id start to id stop.
    hist_d_merged : list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated detour length between primary and each secondary.
        Contains only entries from id start to id stop.
    pop_normal_merged : list of list
        List of histograms per scenario. The histogram contains the count
        of reachable population per timeslot per scenario for each node.
        Contains only entries from id start to id stop.

    """
    secondaryGraphsLength = len(secondaryGraphs)

    pop_normal = {}
    pop_filtered = {i: {} for i in range(0, secondaryGraphsLength)}

    hist_n = {i: {'inf': 0} for i in range(0, secondaryGraphsLength)}
    hist_f = {i: {'inf': 0} for i in range(0, secondaryGraphsLength)}
    hist_d = {i: {'inf': 0} for i in range(0, secondaryGraphsLength)}

    osmids_g = G.vs['osmid']
    paths_filtered = [[] for i in range(0, secondaryGraphsLength)]

    nodes_in_range = []
    # if stop >= len(G.vs()):
    #     stop = stop - 1
    for i in range(start, stop+1):
        nodes_in_range.append(i)

    selected_g = G.vs.select(nodes_in_range)

    paths_normal = G.shortest_paths(source=selected_g,
                                    weights=weights, mode="all")
    osmids = pd.Series([vs["osmid"] for vs in selected_g])
    df_normal = pd.DataFrame(paths_normal)
    df_normal.set_index(osmids, inplace=True)
    df_normal.columns = osmids_g

    for current_osmid in osmids:
        paths_normal_v = df_normal.loc[current_osmid]
        pop_normal[current_osmid] = get_reachable_pops(G, paths_normal_v,
                                                       traveltimes)

    for j in range(0, secondaryGraphsLength):
        H = secondaryGraphs[j]
        paths_filtered[j] = H.shortest_paths(source=selected_g,
                                             weights=weights, mode="all")
        osmids_h = H.vs['osmid']
        df_filtered = pd.DataFrame(paths_filtered[j])
        df_filtered.set_index(osmids, inplace=True)
        df_filtered.columns = osmids_h

        for current_osmid in osmids:
            paths_filtered_v = df_filtered.loc[current_osmid]
            paths_normal_v = df_normal.loc[current_osmid]

            pop_filtered[j][current_osmid] = get_reachable_pops(
                H, paths_filtered_v,
                traveltimes)
            hist_n[j], hist_f[j], hist_d[j] = append_path_stats(
                paths_normal_v, paths_filtered_v,
                hist_n[j], hist_f[j], hist_d[j])

    return (hist_n, hist_f, hist_d, pop_filtered)


def calc_max_nodes_scenario(cfg, city, scenario):
    """Get the amount of nodes in the Graph.

    Parameters
    ----------
    city : str
        The city for which to fetch the Graph.
    scenario : dict
        The Scenario for which to filter the Graph.

    Returns
    -------
    int
        Number of nodes in the filtered Graph.
    """
    graph = oh.get_place(city, simplify=True,
                         file_identifier="bicycle-tags",
                         force_download=False)

    # filter_graph by scenario
    scenario_filter = get_scenario_filter(cfg, scenario["name"])

    H_nx = oh.filter_graph_by_dict(graph, scenario_filter, drop_nodes=True)
    return H_nx.number_of_nodes()


def calc_key_numbers(cfg, override=False):
    """Calculate all key numbers.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    override : bool, default=False
        if true does not use old files to speed up calculation.

    Returns
    -------
    key_numbers : list
        List of all key numbers in all city.
    """
    key_numbers = []
    for counter, city in enumerate(cfr.get_cities(cfg)):

        key_values_city = key_numbers_city(cfg, city, override=override)
        key_values_city = add_markers_and_color(cfg, key_values_city, counter)
        key_numbers.extend(key_values_city)

    return key_numbers


def calc_key_numbers_city(cfg, city, path_key_numbers):
    """Calculate the key numbers for a city.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        City for which to calculate the key numbers.
    path_key_numbers : Path
        Output path were the results will be written to.
    """
    k = []
    pop_total = get_max_pop(cfg, city, "total")
    for scenario in cfr.yield_scenarios(cfg):
        key_number_dict = calc_key_numbers_scenario(cfg, city, scenario,
                                                    pop_total=pop_total)
        key_number_dict["city"] = city
        key_number_dict["scenario"] = scenario
        key_number_dict["scenario_name"] = scenario["name"]
        k.append(key_number_dict)
    df = pd.DataFrame(k)
    df.to_csv(path_key_numbers)


def calc_key_numbers_scenario(cfg, city, scenario, pop_total=0):
    """Calc key numbers for scenario in city.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    scenario : dict
        The scenario for which to do the calculations.
    pop_total : int, optional
        maximum population reachable in the city, by default 0

    Returns
    -------
    key_numbers
        Dictionary with all the key numbers calculated.
    """

    hist_n, hist_f, hist_d, pop_n = load_histograms(cfg, city, scenario)

    df_n = pd.DataFrame(hist_n, index=[0])
    df_f = pd.DataFrame(hist_f, index=[0])
    df_d = pd.DataFrame(hist_d, index=[0])

    max_nodes_scenario = calc_max_nodes_scenario(cfg, city, scenario)
    max_nodes_total = calc_max_nodes_total(df_n)

    ways_filtered = calc_reachable_ways(df_f)

    average_nodes_reachable = ways_filtered / (max_nodes_scenario - 1)

    detour = avg_detour_percentage(df_d)

    comp_stats = load_component_stats(city, scenario)

    scenario_max = get_max_pop(cfg, city, scenario["name"])

    pop_stats = load_pop_stats(pop_n, pop_total, scenario_max)

    # print(comp_stats)

    key_numbers = {"max_ways": ways_filtered,
                   "max_nodes": max_nodes_total,
                   "max_nodes_scenario": max_nodes_scenario,
                   "max_nodes_scenario_percentage": max_nodes_scenario/max_nodes_total,
                   "average_nodes_reachable": average_nodes_reachable,
                   "average_nodes_reachable_percentage_all": average_nodes_reachable/max_nodes_total,
                   "average_nodes_reachable_percentage_scenario": average_nodes_reachable/max_nodes_scenario,
                   "biggest_component_coverage": comp_stats["biggest_component_coverage"] / 100,
                   "component_count": comp_stats["component_count"],
                   "component_count_norm": comp_stats["component_count"] / max_nodes_total,
                   "detour": detour,
                   "pop_total": pop_total,
                   "scenario_max": scenario_max}

    key_numbers.update(pop_stats)

    return key_numbers


def calc_max_nodes_total(df):
    """Get the amount of nodes in the scenario."""
    return df["self"].iloc[0]


def calc_reachable_ways(df, drop_inf=True, drop_self=True):
    """Get the amount of calculated shortest paths.

    Parameters
    ----------
    df : DataFrame
        DataFrame with Histogram of pathlengths.
    drop_inf : bool, optional
        Should shortest paths with length "inf" be dropped, by default True
    drop_self : bool, optional
        should shortest paths with length 0 be dropped, by default True

    Returns
    -------
    int
        The number of valid shortest paths.
    """
    df2 = df
    if drop_inf:
        df2 = df2.drop(["inf"], axis=1, errors="ignore")
    if drop_self:
        df2 = df2.drop(["self"], axis=1, errors="ignore")

    return df2.T.sum().iloc[0]


def dict_pop(length, values):
    """Make Dict of pop values for the time slot. 

    Parameters
    ----------
    length : int
        timeslot for which to generate pop values.
    values : array
        Array of values.

    Returns
    -------
    dict
        Dictionary of pop values with corresponding key for timeslot.
    """
    return {
        "h{}".format(length): values[0],
        "h{}m".format(length): values[1],
        "h{}mn".format(length): values[2],
        "h{}ms".format(length): values[3],
        "h{}n".format(length): values[4],
        "h{}s".format(length): values[5]
    }


def get_node_count_city(cfg, city, c_city, c_scenario, counts):
    """Count nodes per scenario in a city.
    Log the different nodes counts for the scenarios. 

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    c_city : list
        List of done cities.
    c_scenario : list
        List of done scenarios.
    counts : list
        List of counts

    Returns
    -------
    c_city : list
        List of done cities.
    c_scenario : list
        List of done scenarios.
    counts : list
        List of counts
    """
    cc_city = []
    cc_scenario = []
    cc_counts = []
    for scenario in cfr.yield_scenarios(cfg):
        c_city.append(city)
        cc_city.append(city)
        c_scenario.append(scenario)
        cc_scenario.append(scenario)
        count = calc_max_nodes_scenario(cfg, city, scenario)
        counts.append(count)
        cc_counts.append(count)
    df = pd.DataFrame(columns=["cities", "scenarios", "node_count"])
    df["cities"] = cc_city
    df["scenarios"] = cc_scenario
    df["node_count"] = cc_counts
    df.to_csv(Path("out", "node_count_{}.csv".format(city)))
    return c_city, c_scenario, counts


def get_node_counts(cfg):
    """Count nodes for all Cities.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    """

    counts = []
    cities = cfr.get_cities(cfg)
    c_city = []
    c_scenario = []
    for city in cities:
        c_city, c_scenario, counts = get_node_count_city(
            cfg, city, c_city, c_scenario, counts)
        # break
    df = pd.DataFrame(columns=["cities", "scenarios", "node_count"])
    df["cities"] = c_city
    df["scenarios"] = c_scenario
    df["node_count"] = counts
    df.to_csv(Path("out", "node_count.csv"))


def get_graph(place="Freiberg, Sachsen",
              simplify=True,
              clean_periphery=True,
              file_identifier="bicycle-tags",
              got_pop_data=False,
              which_result=0):
    """Load Graph with population assigned.

    Parameters
    ----------
    place : str, default="Freiberg, Sachsen"
        The placename for which to get the Graph.
    simplify : bool, optional
        osmnx parameter,  
        if True, simplify graph topology with the simplify_graph function,
        by default True
    clean_periphery : bool, optional
        osmnx parameter, 
        if True, buffer 500m to get a graph larger than requested,
        then simplify, then truncate it to requested spatial boundaries,
        by default True
    file_identifier : str, optional
        Identifier under which to safe the Graph, by default "bicycle-tags"
    got_pop_data : bool, optional
        if True assign population to the graph with get_pop_per_osmid,
        by default False

    Returns
    -------
    G_nx : networkxGraph
        The Graph of the place requested.
    """

    G_nx = oh.get_place(placename=place, simplify=simplify,
                        clean_periphery=clean_periphery,
                        file_identifier=file_identifier, retain_all=True,
                        which_result=which_result)

    if got_pop_data:
        df_pop = ep.get_pop_per_osmid(place=place, logger=None, simplify=simplify,
                                      file_identifier=file_identifier)
        df_tmp = df_pop["einwohner"]
        df_tmp.index = df_pop["osmid"]
        x = df_tmp.to_dict()
        nx.set_node_attributes(G_nx, x, name="pop")
    else:
        nx.set_node_attributes(G_nx, 1, name="pop")

    print("Done Loading Graph...")

    return G_nx


def get_max_pop(cfg, city, scenario_name, which_result=0):
    """Get the Population assigned to nodes with at least one edge.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    scenario_name : str
        Name of the scenario for which to do the calculations.

    Returns
    -------
    pop_sum : float
        The Population count.
    """

    if scenario_name == "total":
        return ep.get_pop_total(city)

    else:
        scenario_filter = get_scenario_filter(cfg, scenario_name)

        G_nx = get_graph(city, got_pop_data=True, which_result=which_result)
        H_nx = oh.filter_graph_by_dict(G_nx, scenario_filter, drop_nodes=False)

        remove = [node for node, degree in dict(
            H_nx.degree()).items() if degree < 1]
        H_nx.remove_nodes_from(remove)

        pops = dict(H_nx.nodes(data='pop', default=0))

        pop_sum = 0

        for key, value in pops.items():
            if value > 0:
                pop_sum += value

        return pop_sum


def get_max_speed(cfg, edge, speeds={}):
    """Calculate the maximum allowed speed for an edge.
    FlTl

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    edge : dict
        dictionary with osm Tags of that edge.
    speeds : dict, optional
        allowed speeds to drive per osm tag, by default {}

    Returns
    -------
    speed : int
        the allowed maximum speed on the edge.
    """
    # example {'osmid': 26524012, 'name': 'Gailnauer Straße',
    #  'highway': 'unclassified', 'oneway': False,
    #  'length': 31.622999999999998,
    # 'geometry': <shapely.geometry.linestring.LineString object at 0x000001E6D0DC1100>}
    filters = cfr.get_filters(cfg)
    speed = 0
    max_speeds = []
    for filter_name, filter_values in filters.items():
        allowed_speed = speeds.get(filter_name.lower(), 0)
        if allowed_speed <= 0:
            continue
        for tag, allowed_values in filter_values.items():
            if tag in edge:
                if edge[tag] in allowed_values:
                    max_speeds.append(allowed_speed)

    if speeds["overrides"]:
        for override_name, allowed_speed in speeds["overrides"].items():
            filter = cfr.get_override_filter(cfg, override_name.capitalize())
            for key, value in filter.items():
                # print(key, value)
                if key in edge:
                    if edge[key] in value:
                        speed = allowed_speed

    if speed == 0 and max_speeds:
        speed = max(max_speeds)
        # print(max_speeds)
    return speed


def get_pop_sum(G, allowed_nodes):
    """Sums the population in the Graph over all allowed Nodes. 

    Parameters
    ----------
    G : Graph
        Graph with population
    allowed_nodes : set
        Set of allowed nodes to add over.

    Returns
    -------
    pop_sum : float
        sum of population reachable
    """
    pop_sum = 0
    if not allowed_nodes:
        return pop_sum
    for vertex in G.vs():
        if not (vertex["osmid"] in allowed_nodes):
            continue

        if vertex["pop"] > 0:
            pop_sum += vertex["pop"]

    return pop_sum


def get_reachable_pops(G, paths, distances=[1000]):
    """Calculates the reachable population from a single node 
    within maximum distances.

    Parameters
    ----------
    G : Graph
        Graph with assign population.
    paths : numpy Matrix
        Matrix of calculated shortest path distances between pairs of nodes.
    distances : list, optional
        List of maximum distances for which to calculate
        the maximum population reachable, by default [1000]

    Returns
    -------
    pops : list
        list of reachable populations from that node for each maximum distance.
    """

    pops = []

    for length_max in distances:
        df_length = paths[(paths > 0) & (paths < length_max)]
        pops.append(get_pop_sum(G, set(df_length.index)))

    return pops


def get_scenario_filter(cfg, name, speeds=None):
    """Build a dictionary that can be used as a filter.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    name : str
        the name of the filter to fetch from the config.
    speeds : dict, optional
        dictionary of speeds per category i.e. "Intercity:10",
        by default None, will be resolved by cfr.get_speeds(cfg, name)

    Returns
    -------
    d: dict
        dictionary of tags that are used.

    Example
    -------
    {'living_street': ['yes'],
    'highway': ['footway', 'path', 'living_street', 'pedestrian'],
    'foot': ['designated', 'yes', 'permissive'], ...}
    """
    list_filters = []
    if speeds is None:
        speeds = cfr.get_speeds(cfg, name)

    for key, value in speeds.items():
        if key == "overrides":
            for entry, _ in value.items():
                # print(entry)
                e = "{}".format(entry)
                list_filters.append({"highway": [e], e: ["yes"]})
        elif value > 0:
            list_filters.append(cfr.get_filter_params(cfg, key.title()))

    d = merge_dicts_extend_list(list_filters)
    return d


def jsonKeys2int(x):
    """Convert json keys to int"""
    if isinstance(x, dict):
        x = {(int(k) if (k != "inf" and k != "self" and k !=
              "controll") else k): v for k, v in x.items()}
    return x


def key_numbers_city(cfg, city, override=False):
    """Get the key numbers for a city.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    override : bool, default=False
        if true does not use old files to speed up calculation.

    Returns
    -------
    key_numbers : dict
        all important key numbers calculated.

    """
    path_key_numbers = Path("out", "key_numbers_{}_save.csv".format(city))
    if not path_key_numbers.is_file() or override:
        k = calc_key_numbers_city(cfg, city, path_key_numbers)
    k = load_key_numbers_city(path_key_numbers)
    return k


def largest_connected_component(cfg, city, which_result=0):
    """Calculate the largest components of the city.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    city : str
        Name of the city for which to do the calculations.
    """
    df = pd.DataFrame()

    results = []
    G_nx = get_graph(city, got_pop_data=True, which_result=which_result)

    for sz in cfr.yield_scenarios(cfg):

        scenario_filter = get_scenario_filter(cfg, sz["name"])

        nodes = G_nx.number_of_nodes()
        print("copy")
        H_nx = oh.filter_graph_by_dict(
            G_nx.copy(), scenario_filter, drop_nodes=False)
        print("copy done.")
        largest_cc = max(nx.weakly_connected_components(H_nx), key=len)
        print(len(largest_cc))

        coverage_largest_component = round((len(largest_cc) / nodes) * 100, 4)

        print("The largest component can reach {}%".format(
            coverage_largest_component))

        component_count = nx.number_weakly_connected_components(H_nx)

        results.append(
            (city, sz["name"], coverage_largest_component, component_count))

    df = pd.DataFrame(results)
    print(df)

    df.to_csv(Path("out", "{}_components.csv".format(city)))


def load_component_stats(city, scenario):
    """Read the component stats from csv."""
    file = Path("out", "{}_components.csv".format(city))
    df = pd.read_csv(file)
    df.columns = ("counter", "cities", "scenario",
                  "biggest_component_coverage", "component_count")
    return df[df["scenario"] == scenario["name"]].iloc[0]


def load_histograms(cfg, city, scenario, folder=None):
    """Load the histograms and pop data for the city and scenario.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the Config.
    city : str, default=""
        city to load.
    scenario : scenario
        scenario to load.
    folder : str, default=None
        Outputfolder location. Default cfg["Folders"]["out"]

    Returns
    ----------
    hist_n : dict
        Dictionary of how often a path length 
        was calculated in the primary Graph.
    hist_f : dict
        Dictionary of how often a path length 
        was calculated in the secondary Graph.
    hist_d : dict
        Dictionary of how often a detour was calculated.
    pop_n : dict
        Dictionary of reachable pop count in given timeslots per osmid.
    """

    if not city:
        raise ValueError("No City given.")

    if not scenario:
        raise ValueError("No scenario given.")

    if folder is None:
        folder = cfg["Folders"]["out"]

    file_hist_n = Path(folder, city + "_" + scenario["name"] + "_hist_n.json")
    file_hist_f = Path(folder, city + "_" + scenario["name"] + "_hist_f.json")
    file_hist_d = Path(folder, city + "_" + scenario["name"] + "_hist_d.json")
    file_pop = Path(folder, city + "_" + scenario["name"] + "_pop.json")

    if (file_hist_n.is_file()
            and file_hist_f.is_file()
            and file_hist_d.is_file()
            and file_pop.is_file()):
        hist_n = dict(json.load(open(file_hist_n), object_hook=jsonKeys2int))
        hist_f = dict(json.load(open(file_hist_f), object_hook=jsonKeys2int))
        hist_d = dict(json.load(open(file_hist_d), object_hook=jsonKeys2int))
        pop_n = dict(json.load(open(file_pop), object_hook=jsonKeys2int))
        return hist_n, hist_f, hist_d, pop_n
    else:
        print("Didn't have these files:")
        print(file_hist_n, file_hist_n.is_file())
        print(file_hist_f, file_hist_f.is_file())
        print(file_hist_d, file_hist_d.is_file())
        print(file_pop, file_pop.is_file())
        print("Make sure to run the calculations first.")
        return None


def load_key_numbers_city(path_key_numbers):
    """Load the key numbers from file."""
    df = pd.read_csv(path_key_numbers, header=0, index_col=0)
    k = df.to_dict(orient="records")
    return k


def load_pop_scenario(cfg, graph, city,
                      scenario_name, simplify=True, override=False):
    """Load the population data into a graph.

    Parameters
    ----------
    cfg : ConfigObj
        Instance of the config file.
    graph : [type]
        [description]
    city : str
        Name of the city for which to do the calculations.
    scenario_name : str
        Name of the scenario to load.
    simplify : bool, optional
        osmnx parameter,  
        if True, simplify graph topology with the simplify_graph function,
        by default True
    override : bool, default=False
        if true does not use old files to speed up calculation.

    Returns
    -------
    graph
        Graph with population data per node
    """
    if simplify:
        simple = "simple"
    else:
        simple = "complex"

    file_string = "{}_gridpop_{}_{}.csv".format(city, simple, scenario_name)

    grid_pop_file = Path("csvs", file_string)

    if grid_pop_file.is_file() and not override:
        df_pop = pd.read_csv(str(grid_pop_file))
    else:
        city_csv = r"csvs/{}-10m.csv".format(city)

        scenario_filter = get_scenario_filter(cfg, scenario_name)
        filtered_graph = graph.copy()
        filtered_graph = oh.filter_graph_by_dict(
            filtered_graph, scenario_filter, drop_nodes=True)
        df_pop = ep.assign_pop(filtered_graph, city_csv, grid_pop_file)

    df_tmp = df_pop["einwohner"]
    df_tmp.index = df_pop["osmid"]
    x = df_tmp.to_dict()
    nx.set_node_attributes(graph, x, name="pop")

    return graph


def load_pop_stats(pop_n, pop_total, scenario_max,
                   lengths=[60, 300, 600, 1200, 1800, 3600, 360000]):
    """Calculate different statistics for the population data.

    Parameters
    ----------
    pop_n : dict
        Dictionary of reachable pop count in given timeslots per osmid.
    pop_total : float
        maximum population in graph
    scenario_max : float
        maximum population in filtered graph
    lengths : list, optional
        lengths for which to filter, 
        by default [60, 300, 600, 1200, 1800, 3600, 360000]

    Returns
    -------
    dict
        Population data per timeslot.
    """
    df = pd.DataFrame(pop_n)
    df.index = lengths
    df = df.T.reset_index(drop=True)
    length = len(df.index)
    maxs = df.max()
    dfs = df.sum() / length
    x = {}
    for i, v in enumerate(lengths):
        values = [
            dfs.iloc[i],
            maxs.iloc[i],
            maxs.iloc[i] / pop_total,
            maxs.iloc[i] / scenario_max,
            dfs.iloc[i]/pop_total,
            dfs.iloc[i]/scenario_max
        ]
        x.update(dict_pop(v, values))

    return x


def merge_dicts_extend_list(list_of_dicts):
    """Merge dictionaries of list. List are extended."""
    dict_merged = {}
    all_keys = set()

    for dict_i in list_of_dicts:
        all_keys = all_keys | set(dict_i)

    for key in all_keys:
        for dict_i in list_of_dicts:
            value = dict_i.get(key, [])
            if value:
                current = dict_merged.get(key, [])
                if value != current:
                    dict_merged[key] = list(set(current) | set(value))

    return dict_merged


def merge_dicts_add_counts(list_of_dicts):
    """Merge dictionaries of values. Values are added."""
    dict_merged = {}
    all_keys = set()
    for dict_i in list_of_dicts:
        all_keys = all_keys | set(dict_i)

    for key in all_keys:
        key_sum = 0
        for dict_i in list_of_dicts:
            key_sum += dict_i.get(key, 0)

        dict_merged[key] = key_sum

    return dict_merged


def merge_dicts_list_add_counts(list_of_dicts):
    """Merge dictionaries of lists. Values are added per index."""
    dict_merged = {}
    all_keys = set()
    for dict_i in list_of_dicts:
        all_keys = all_keys | set(dict_i)

    # get the length of the arrays that are to be merged
    len_of_list = len(list_of_dicts[0][list(list_of_dicts[0])[0]])

    for key in all_keys:
        key_sum = [0 for i in range(len_of_list)]
        for dict_i in list_of_dicts:
            # print(dict_i)
            list_pop = dict_i.get(key, [0 for i in range(len_of_list)])
            # print(list_pop)
            counter = 0
            for pop in list_pop:
                key_sum[counter] += pop
                counter += 1

        dict_merged[key] = key_sum

    return dict_merged


def prep_parallel_assign(cfg, G_nx, city,
                         simplify, override, with_speed):
    """Prepare List of Tupels for parallel assigning of pop."""
    values = ()
    for scenario in cfr.yield_scenarios(cfg):
        values = values + ((cfg, G_nx.copy(), city,
                            scenario, simplify, override,
                            with_speed),)
    return values


def prep_parallel_runtime_optimized(cfg, G_ig, secondaryGraphs):
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    n = 0
    cores = cfg["Parallel"].as_int("cores")
    chunk_size = cfg["Parallel"].as_int("chunk_size")
    chunk_size_max = cfg["Parallel"].as_int("chunk_size_max")
    max_mem = cfg["Parallel"].as_float("max_mem")
    traveltimes = cfr.get_traveltimes(cfg)
    secondaryGraphsLength = len(secondaryGraphs)
    print(type(G_ig))
    length = len(G_ig.vs())
    if n > length or n == 0:
        n = length

    if cores is None or cores == 0:
        cores = cpu_count()
    if cores < 1:
        cores = 1

    print(f'\nStarting computations on {cores} of {cpu_count()} cores.')

    if chunk_size == 0:
        chunk_size, leftover = divmod(n, cores)

    node_count = len(G_ig.vs())

    # we need to fit all the processes in the available ram...
    try:
        mem_graph = actualsize(G_ig)
    except Exception as e:
        print(e)
        mem_graph = node_count * 64

    fit_mem = 100
    print("Estimating Ram Usage...")
    while chunk_size > chunk_size_max or fit_mem > max_mem:
        chunk_size, leftover = divmod(chunk_size, 2)
        mem_available = psutil.virtual_memory().available
        """Each chunk will calc all paths from starting range to all other nodes.
            To save the result we need an float each. 
        """
        mem_chunk = chunk_size * node_count * 24
        # All Graphs = the primary + the secondaries.
        # Secondaries are smaller than primary
        # Each core will have a copy of the graphs (at least thats what it looks like)
        mem_graphs = mem_graph * (secondaryGraphsLength + 1) * (cores + 1)

        chunk_count = n/chunk_size

        mem_estm = (mem_graphs
                    + (mem_chunk * chunk_count)
                    + (mem_chunk * leftover))

        fit_mem = mem_estm/mem_available
        print("""\
            Mem per chunk: \t{},\n\
            mem graphs \t{},\n\
            mem total \t{},\n\
            mem available \t{},\n\
            estm % used \t{} going for < {}%\n""".format(
            (mem_chunk*chunk_count)/1000000000,
            mem_graphs/1000000000,
            mem_estm/1000000000,
            mem_available/1000000000,
            fit_mem*100,
            max_mem*100))

    if chunk_size < 1:
        raise Exception("""
            Chunk size is to small (<1). 
            Do you have enough memory to run this graph?\n\
            Needed: {} Free: {} ; {}% """.format(
            mem_estm, mem_available, fit_mem*100))

    values = ()
    values = values + ((G_ig, secondaryGraphs, 0, chunk_size, traveltimes),)

    upper_bound = 0
    i = 0
    last_chunk = n - chunk_size

    while upper_bound < last_chunk:
        i += 1
        lower_bound = i*chunk_size + 1
        upper_bound = (i+1)*chunk_size
        values = values + \
            ((G_ig, secondaryGraphs, lower_bound, upper_bound, traveltimes),)

    if upper_bound < n - 1:
        values = values + \
            ((G_ig, secondaryGraphs, upper_bound + 1, n-1, traveltimes),)

    print("Calculating {} chunks with avg. size {} on {} cores. \n\
          Size per chunk: {} Estm Ram Usage: {}%".format(
        len(values), chunk_size, cores, mem_chunk, int(fit_mem*100)))

    return values, cores


def run_parallel_runtime_optimized(values, cores, scenario_count=5):
    """Preperation for the parallel computation start.

    Parameters
    ----------
    values : Tupel
        Prepared Values to pass via starmap. See prep_parallel().
    cores : int
        Cores allowed to use for parallel computation.
    scenario_count : int, optional
        amount of scenarios that are going to be run, by default 5

    Returns
    -------
    hist_n_merged: list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated shoretest path length in the primary graph.
    hist_f_merged: list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated shoretest path length in the secondary graphs.   
    hist_d_merged: list of list
        List of histograms per scenario. The histogram contains the count
        of the calculated detour length between primary and each secondary.
    pop_normal_merged: list of list
        List of histograms per scenario. The histogram contains the count
        of reachable population per timeslot per scenario for each node.
    """
    hist_n_list = [[] for i in range(scenario_count)]
    hist_f_list = [[] for i in range(scenario_count)]
    hist_d_list = [[] for i in range(scenario_count)]
    pop_normal_list = [[] for i in range(scenario_count)]

    hist_n_merged = [[] for i in range(scenario_count)]
    hist_f_merged = [[] for i in range(scenario_count)]
    hist_d_merged = [[] for i in range(scenario_count)]
    pop_normal_merged = [[] for i in range(scenario_count)]

    for value in values:
        print(value)

    with Pool(cores) as pool:

        res = pool.starmap(calc_scenarios_runtime_optimized,
                           tqdm.tqdm(values, total=len(values)))

        for entry in res:
            for j in range(0, scenario_count):
                hist_n_list[j].append(entry[0][j])
                hist_f_list[j].append(entry[1][j])
                hist_d_list[j].append(entry[2][j])
                pop_normal_list[j].append(entry[3][j])

        for j in range(0, scenario_count):
            hist_n_merged[j] = merge_dicts_add_counts(hist_n_list[j])
            hist_f_merged[j] = merge_dicts_add_counts(hist_f_list[j])
            hist_d_merged[j] = merge_dicts_add_counts(hist_d_list[j])

            pop_normal_merged[j] = merge_dicts_list_add_counts(
                pop_normal_list[j])

    return hist_n_merged, hist_f_merged, hist_d_merged, pop_normal_merged


def assign_speed(cfg, G_nx, G_ig, scenario_name):
    """Assign allowed max speed to edges."""
    # TODO this can probably only use G_ig.

    speeds = cfr.get_speeds(cfg, scenario_name)
    for u, v, k, edge in G_nx.edges(keys=True, data=True):
        max_speed = get_max_speed(cfg, edge, speeds)

        if max_speed == 0:
            edge['length'] = float('inf')
        else:
            edge['length'] = edge['length'] / (max_speed / 3.6)

    G_ig.es["length"] = list(nx.get_edge_attributes(G_nx, "length").values())

    return G_ig


def nx_to_ig(cfg, G_nx, scenario_name=None, with_speed=True):
    """Convert a networkx graph to an python igraph.

    Parameters
    ----------
    G_nx : networkxGraph
        The Input Graph
    cfg : ConfigObj
        Instance of the config file.
    scenario_name : [type], optional
        scenario for which to convert. Is needed if speed with_speed is True,
        by default None
    with_speed : bool, optional
        if True the edge length will be divided by allowed max speed,
        by default True

    Returns
    -------
    G_ig : python igraph    
        Converted python igraph with speed.
    """

    osmids = list(G_nx.nodes)
    G_nx = nx.relabel.convert_node_labels_to_integers(G_nx)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k: v for k, v in zip(G_nx.nodes, osmids)}
    nx.set_node_attributes(G_nx, osmid_values, 'osmid')

    # convert networkx graph to igraph
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(G_nx.nodes)
    G_ig.add_edges(G_nx.edges())
    G_ig.vs['osmid'] = osmids
    G_ig.vs['pop'] = list(nx.get_node_attributes(G_nx, "pop").values())

    if with_speed:
        G_ig = assign_speed(cfg, G_nx, G_ig, scenario_name)

    return G_ig


def main(cfg):
    """

    """
    cities = cfr.get_cities(cfg)
    override = cfr.get_override(cfg)
    simplify = cfr.get_simplify(cfg)
    which_results = cfr.get_which_results(cfg)  # TODO default to 0

    # ox.config(useful_tags_way=cfr.get_relevant_tags(cfg))
    start = timer()
    for city, which_result in zip(cities, which_results):
        start_city = timer()
        ep.get_population_data(
            city, file_identifier="bicycle-tags",
            useful_tags_way=cfr.get_relevant_tags(cfg),
            which_result=which_result)
        analyse_runtime_optimized(
            cfg, city, override=override, with_speed=True,
            simplify=simplify, which_result=which_result)

        largest_connected_component(cfg, city, which_result=which_result)
        get_node_count_city(cfg, city, [], [], [])

        end_city = timer()
        print(f'elapsed time: {end_city - start_city}')

    get_node_counts(cfg)
    calc_key_numbers(cfg, override=override)
    end = timer()
    print(f'elapsed time: {end - start}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="Name or absolute Path of a .ini file to use.",
        default="ready4robots.ini")

    args = parser.parse_args()
    cfg = ConfigObj(args.config)

    ox.config(useful_tags_way=cfr.get_relevant_tags(cfg))
    ox.config(use_cache=True, log_console=True)

    main(cfg)
