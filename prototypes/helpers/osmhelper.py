# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:23:16 2020

@author: plank
"""

import osmnx as ox
from pathlib import Path
import numpy as np
import networkx as nx

import os
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd

def determineOSMID(u, v, edges):
    x = edges.osmid[(edges.u == u) & (edges.v == v)].iloc[0]

    if x :
        return x
    else:
        return 0

def determine_edge_id(u, v, edges):
    x = edges.osmid[(edges.u == u) & (edges.v == v)].index.to_list()[0]
    if x :
        return x
    else:
        return 0

def calc_track_osmids(track, graph, edges):
    """Faster way to calculate all the osmids of all edges that we want to match."""
    np_track = track.to_numpy()
    ids = []
    x = np_track[:, 3]
    y = np_track[:, 4]
    results = ox.get_nearest_edges(graph, y, x, method = 'kdtree')

    #todo there may be multible edges. how to handle that?
    for result in results:
        u = result[0]
        v = result[1]

        x = edges.osmid[(edges.u == u) & (edges.v == v)].index.to_list()[0]
        if x :
            ids.append(x)
        else:
            ids.append(0)

    return ids



def make_logger():
    
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/paper.log',
                                       maxBytes=131072, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s]: %(message)s '
        '[in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    
    logger.info("Created Logger.")
    
    
    return logger

def get_place_filtered(placename, graph_paths, simplify = False,
              filter_tags = [], force_download = True, file_identifier = None,
              logger = None, clean_periphery = True):

    if logger is None:
        logger = logging.getLogger()

    logger.info("test")

    if filter_tags and file_identifier is None:
        force_download = True


    graph, file_graph = load_graph_disk(
        placename, graph_paths = graph_paths, simplify = simplify,
        force_download = force_download, file_identifier = file_identifier, logger=logger)

    if graph is not None:
        return graph

    graph_list = []

    for tag in filter_tags:
        logger.info("{}, {}".format(tag, placename))

        try:
            tmp_graph = ox.graph_from_place(
                placename, network_type=None,
                simplify = simplify,
                retain_all=True,
                custom_filter='["{0}"]'.format(tag),
                clean_periphery = clean_periphery)
               # custom_filter=tag)
            if tmp_graph:
                graph_list.append(tmp_graph)
            else:
                logger.info("No Elements.")

        except Exception as e:
            logger.info(e)

    if graph_list:
        graph = nx.compose_all(graph_list)

        logger.info("{}, {}".format(placename, file_graph))

        ox.save_graphml(graph, file_graph)

        return graph

def load_graph_disk(placename, graph_paths, simplify, force_download, file_identifier, logger = None):
    for graph_path in graph_paths:
        if simplify:
            if file_identifier is None or file_identifier == "":
                file_graph = Path(graph_path, placename)
            else:
                file_graph = Path(graph_path, placename + "_" + file_identifier)
        else:
            if file_identifier is None or file_identifier == "":
                file_graph = Path(graph_path, placename + "_complex")
            else:
                file_graph = Path(graph_path, placename + "_" + file_identifier + "_complex")
        logger.info("{}".format(file_graph))

        if file_graph.is_file() and not force_download:
            logger.info("Loading from disk: {}".format(placename))

            graph = ox.load_graphml(file_graph)
            return graph, file_graph
        else:
            return None, file_graph


def get_place(placename, graph_paths = ["osm"], simplify = True,
              network_type="all_private",
              force_download = False, file_identifier = None,
              logger = None, clean_periphery = True,
              retain_all = True, which_result=0):

    if logger is None:
        logger = logging.getLogger()
        
    logger.info("{}".format(placename))

    df = ox.geocoder.geocode_to_gdf(placename, which_result)
    #TODO better filter for big sets...
        #Sometimes the data is just to large or unresonable.
    diagonal = ox.distance.euclidean_dist_vec(
            df.bbox_north[0], df.bbox_east[0], df.bbox_south[0], df.bbox_west[0])
    logger.info("{}".format(diagonal))

    graph, file_graph = load_graph_disk(
        placename, graph_paths = graph_paths, simplify = simplify,
        force_download = force_download, file_identifier = file_identifier, logger=logger)

    if graph is not None:
        return graph

    if diagonal < 1:

        try:
            graph = ox.graph_from_place(
                placename, network_type=network_type,
                simplify = simplify,
                clean_periphery = clean_periphery,
                retain_all = True)
            logger.info("{}, {}".format(placename, file_graph))

            ox.save_graphml(graph, file_graph)
        except Exception as e:
            logger.info(e)
            logger.warning(" {0} did not seem to resolve as Polygon, trying to fetch BoundingBox instead.".format(
                placename))
            try:
                north = df.bbox_north[0]
                east = df.bbox_east[0]
                south = df.bbox_south[0]
                west = df.bbox_west[0]
                graph = ox.graph_from_bbox(north, south, east, west)

                logger.info('\t Got BBox successfully.')
                logger.info("{}, {}".format(placename, file_graph))

                ox.save_graphml(graph, file_graph)
            except Exception as e:
                logger.info(e)

                return None
    else:
        logger.info('Area to big.')

    return graph

def get_place_tags(place = "Freiberg, Sachsen",
                simplify = True,
                clean_periphery = True ,
                file_identifier = "bicycle-tags", 
                got_pop_data = False):
    
    useful_tags_way = ['bridge', 'tunnel', 'oneway', 'lanes', 'ref', 'name',
                   'highway', 'maxspeed', 'service', 'access', 'area',
                   'landuse', 'width', 'est_width', 'junction',
                   'sidewalk', 'cycleway', 'bicycle', 'footway',
                   'cyclestreet', 'path', 'foot', 
                   "sidewalk:right", "sidewalk:left", "sidewalk:both",
                   "cycleway:right", "cycleway:left", "cycleway:both",
                   "width", "surface", "smoothness",
                   "lanes:width", "lanes:surface", "lanes:smoothness",
                   "max_width", "est_width"]

    ox.config(useful_tags_way=useful_tags_way)
    
    G_nx = get_place(placename = place, simplify=simplify,
                     clean_periphery = clean_periphery,
                     file_identifier = file_identifier, retain_all = True)
    
    return G_nx

def match_point_osm(graph, lat, lon):

    (u,v,key, geom) = ox.get_nearest_edge(graph, (lat,lon), return_geom=True)

    tmp_edges = ox.graph_to_gdfs(graph, nodes=False,
                                        node_geometry=False, edges=True)

    osmid = determine_edge_id( u, v, tmp_edges)

    return osmid

def truncate_graph_gnss(graph, df_gnss, logger = None):

    north = df_gnss['lat_matched'].max()
    south = df_gnss['lat_matched'].min()
    west = df_gnss['lon_matched'].min()
    east = df_gnss['lon_matched'].max()

    try:
        tmp_graph = ox.truncate.truncate_graph_bbox(
                            graph, north, south,
                            east, west, truncate_by_edge = True, retain_all=True)
        return tmp_graph
    except Exception as e:
        logger.info(e)
        logger.error("Graph:{0} nsew:{1},{2},{3},{4}".format(
            graph, north, south, east, west))

def match_points_osm(graph, df_gnss, logger = None):

    if logger is None:
        return


    ne = ox.get_nearest_edges(graph,
                            df_gnss['lon_matched'],
                            df_gnss['lat_matched'],
                            method='balltree')

    logger.info("Done getting nearest Edges...")
    logger.info("Assigning osm_ids... This may take a while for big Graphs.")

    df_gnss['u'] = ne[:, 0]
    df_gnss['v'] = ne[:, 1]


    edges = ox.graph_to_gdfs(graph, nodes=False,
                                    node_geometry=False, edges=True)

    df_gnss['osm_id'] = df_gnss.apply(
            lambda x: determineOSMID(
                x.u, x.v, edges
            ), axis=1
        )


    # df_gnss['edge_id'] = df_gnss.apply(
    #         lambda x: determine_edge_id(
    #             x.u, x.v, edges
    #         ), axis=1
    #     )

    return df_gnss


def convertOSMid(edges):
    # Transforms osmid entries into a list
    aux = edges.osmid.to_list()
    for index, entry in enumerate(aux):
        if type(entry)==int:
            aux[index]=[entry]
    edges["osmid"] = aux
    return edges


def determineStreetType(edges, edge_id, logger = None):
    cycleway = "1"
    footway  = "2"
    street   = "3"
    error    = "-1"

    walking_and_cycling = ["cycleway", "footway", "path"]
    streets = ["trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
            "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service"]
    wrong_streets = ["motorway", "motorway_link"]
    sidewalk_options = ["both", "right", "left"]
    cycleway_options = ["lane", "opposite", "opposite_lane", "track", "opposite_track"]
    access_allowed = ["yes", "designated"]

    try:
        edge = edges[edges["osmid"] == edge_id].iloc[0]

    except Exception as e:
        logger.info(e)
        logger.info("{}".format(edges))
        logger.info("{}".format(edge_id))

    edge_hs = edge["highwayString"]
    edge_hsl = len(edge_hs.split(","))

    # im highwayString befindet sich ein Element
    if (edge_hsl == 1):

        # Treppen
        if (edge_hs == "steps"):
            return error

        # kombinierter Geh- und Radweg
        elif ((edge_hs in walking_and_cycling)
            and (edge["bicycle"] == "designated")
            and (edge["foot"]    == "designated")):
            return cycleway, footway

        # eigenständiger Gehweg
        elif (edge_hs == "footway"):
            return footway

        # eigenständiger Radweg
        elif (edge_hs == "cycleway"):
            return cycleway

        # Autobahn
        elif (edge_hs in wrong_streets):
            return error

        # Straßen
        elif (edge_hs in streets):
            sidewalk_available = False
            cycleway_available = False

            # Gehweg vorhanden
            if (edge["sidewalk"] in sidewalk_options):
                sidewalk_available = True

            # Radweg vorhanden
            if (edge["cycleway"] in cycleway_options):
                cycleway_available = True

            if (sidewalk_available and cycleway_available):
                return footway, cycleway, street
            elif (sidewalk_available):
                return footway, street
            elif (cycleway_available):
                return cycleway, street
            else:
                return street

        # Fußgängerzone
        elif (edge_hs == "pedestrian"):
            return footway, street

        # Wald-/Wirtschaftswege für zweispurige Fahrzeuge befahrbar
        elif (edge_hs == "track"):
            return footway, cycleway, street

        # Pfad für nichtmotorisierte Nutzung
        elif (edge_hs == "path"):
            return footway, cycleway

        # unbekannter Straßentyp
        elif (edge_hs == "road"):
            sidewalk_available = False
            cycleway_available = False
            motor_access       = True

            # Gehweg vorhanden
            if (edge["sidewalk"] in sidewalk_options):
                sidewalk_available = True

            # für Fußgänger gewidmet
            elif (edge["foot"] == "designated"):
                sidewalk_available = True

            # Radweg vorhanden
            if (edge["cycleway"] in cycleway_options):
                cycleway_available = True

            # für Radfahrer gewidmet
            elif (edge["bicycle"] == "designated"):
                sidewalk_available = True

            # für Autoverkehr gesperrt
            if (edge["motor_vehicle"] == "no"):
                motor_access = False

            if (sidewalk_available and cycleway_available and motor_access):
                return cycleway, footway, street
            elif (sidewalk_available and cycleway_available):
                return cycleway, footway
            elif (sidewalk_available and motor_access):
                return footway, street
            elif (cycleway_available and motor_access):
                return cycleway, street
            elif (sidewalk_available):
                return footway
            elif (cycleway_available):
                return cycleway
            elif (motor_access):
                return street
            else:
                logger.info("path without driving authorization")
                logger.info("{}, {}".format(edges.osmid[edge_id], edge_hs))

                return error

        # Straßentyp wurde nicht behandelt
        else:
            logger.info("{}".format(edge_hs))
            return error



    # manuelle Fehlerbehandlung bei doppelten Werten
    elif (edge_hsl == 2):
        if ("steps" in edge_hs):
            logger.info("steps found")
            return error

        elif (("footway" in edge_hs) and ("service" in edge_hs)):
            return footway, street

        elif (("path" in edge_hs) and ("service" in edge_hs)):
            return street

        elif (("path" in edge_hs) and ("residential" in edge_hs)):
            return street

        elif (("pedestrian" in edge_hs) and ("footway" in edge_hs)):
            return footway

        elif (("track" in edge_hs) and ("service" in edge_hs)):
            return street

        elif (("pedestrian" in edge_hs) and ("residential" in edge_hs)):
            return footway, street

        elif (("service" in edge_hs) and ("residential" in edge_hs)):
            return street

        elif (("living_street" in edge_hs) and ("path" in edge_hs)):
            return street

        else:
            logger.info("{}, {}".format(edge_hs, edge_id))
            return error

    else:
        if ("steps" in edge_hs):
            logger.info("steps found")
            return error

        # todo Fehlerbehandlung für mehr als 2 Einträge in edge_hs
        return error

        # logger.info(edge_id, edge_hsl)
        # logger.info(edge_hs)


def preprocessOSMData(edges,
    relevant_track_types = ["cycleway", "footway", "path",
                            "pedestrian", "unclassified", "service",
                            "secondary", "residential", "steps",
                            "tertiary", "track", "primary"],
    my_tag_list = ['highway', 'cycleway', 'sidewalk',
                   'bicycle', 'length', 'surface',
                   'smoothness', 'width', 'maxwidth',
                   'est_width', 'foot', 'bicycle',
                   'horse', 'motor_vehicle'],
    logger = None):

    # Add missing columns (if they are not included already)
    for column in my_tag_list:
        if column not in edges.columns:
            edges[column]=np.NaN

    # Perparation of the highway entries (sometimes they combine different categories)
    #if flask:

    edges['highwayString']= edges['highway'].astype(str)
    for column in relevant_track_types:
        if column not in edges.columns:
            edges[column]=edges.highwayString.str.contains(column)
        else:
            logger.info("{}".format(column))

    return edges


def get_filter_tags(street_type, attribute):

    tag_list = """
        {0}:both:{1},
        {0}:right:{1},
        {0}:left:{1},
        {0}:{1},
        {1},
        lanes:{1}

    """.format(street_type, attribute)



    tag_list = tag_list.replace("\n", "").replace(" ", "").split(",")

    return tag_list
    
def count_tags(G, tag = "width"):
    
    count = 0
    
    for u, v, k, d in G.edges(keys=True, data=True):
        if tag in d:
            count +=1

    return count

def stats_tag(G, tag = "width"):
    
    total = len(G.edges())
    
    count = count_tags(G, tag)
    
    return count, total, (count/total)*100

def filter_graph_by_tags(G, tags, drop_nodes = False):
    
    filtered_edges = []
    
    for u, v, k, d in G.edges(keys=True, data=True):
        
        keep = False
        
        for tag in tags:
            if tag in d:
                keep = True
    
        if not keep:
            filtered_edges.append((u, v, k))
             
            
    G.remove_edges_from(filtered_edges)
    
    if drop_nodes == True:
        remove = [node for node,degree in dict(G.degree()).items() if degree < 1]
        G.remove_nodes_from(remove)

    return G



def filter_graph_by_attributes(G, attributes = ['width', 'surface']):
    
    tags = get_all_filter_tags(attributes=attributes)

    return filter_graph_by_tags(G, tags)
    
def filter_graph_by_attribute(G, attribute = "width"):
        
    return filter_graph_by_attributes(G, [attribute])
    
def filter_graph_by_dict(G, filter_dict, drop_nodes = False):
    
    filtered_edges = []
    
    for u, v, k, d in G.edges(keys=True, data=True):
        keep = False
        for key, value in filter_dict.items():
            if key in d:
                if not value is None:
                    #print(d[key])
                    if isinstance(d[key], list):
                        for sub_key in d[key]:
                            if sub_key in value:
                                keep = True
                    else:
                        if d[key] in value:
                            keep = True
                else:
                    keep = True
        if not keep:
            #print(u, v, k)
            filtered_edges.append((u, v, k))
    
    G.remove_edges_from(filtered_edges)
    
    if drop_nodes == True:
        remove = [node for node,degree in dict(G.degree()).items() if degree < 1]
        G.remove_nodes_from(remove)

    return G

def get_all_filter_tags(street_types = ["sidewalk", "cycleway", "footpath"],
                        attributes = ["width"]):

    if isinstance(attributes, list):
        all_tags = []
        for street_type in street_types:
            for attribute in attributes:
                all_tags.extend(get_filter_tags(street_type, attribute))

    else:
        for street_type in street_types:
            all_tags.extend(get_filter_tags(street_type, attribute))
            
    unique_set = set(all_tags)
    unique_tags = list(unique_set)

    return unique_tags

def tags_to_overpass_string(list_of_tags, logger = None):

    logger.info("{}".format(list_of_tags))

    string_of_tags = "\'["

    for tag in list_of_tags:
        string_of_tags+=tag + ";"

    string_of_tags = string_of_tags[:-1]
    string_of_tags += "]\'"

    logger.info("{}".format(string_of_tags))

    return string_of_tags

def get_all(place, graph_paths, logger = None):

    attributes=["width","smoothness","surface"]

    for attribute in attributes:

        filtered_tags = get_all_filter_tags(attribute=attribute)
        logger.info("{}".format(filtered_tags))

        G = get_place_filtered(place, graph_paths=graph_paths, simplify=False,
                      filter_tags=filtered_tags, force_download=True, file_identifier=attribute,
                      logger = logger)
        fmap = ox.plot_graph_folium(G, edge_color = "green", edge_width = 1)

        y = place + "_" + attribute
        fmap.save("{}.html".format(y))


def tag_stats(place = "Freiberg, Sachsen"):
    
    
    # G_ig = nx_to_ig(G_nx, speeds = szenario)
    filter_names = ["intercity", "local", "cycleway", "footway", "path", "sidewalk", "bicycle"]
    tags = ["surface", "width", "smoothness", "max_width", "est_width"]

    df = pd.DataFrame(index = tags)

    for filter_name in filter_names:
        G_nx = get_place_tags(place)
        if G_nx is None:
            return
        H_nx = filter_graph_by_dict(G_nx, get_filter_by_name(filter_name))        
        
        s = "{}\n".format(filter_name)
        tag_stats = []
        for tag in tags:
            count, total, percent = stats_tag(H_nx, tag)
            percent = round(percent, 2)
            s += "{}: {}\n".format(tag, percent)
            tag_stats.append(percent)
            #print(count, total, percent)
        #print(tag_stats)
        df[filter_name] = tag_stats
        
        #print(s)
    print(df)
    

def merge_dicts_list(list_of_dicts):

    dict_merged = {}
    all_keys = set()
    for dict_i in list_of_dicts:
        all_keys = all_keys | set(dict_i)
    
    for key in all_keys:
        key_sum = []
        for dict_i in list_of_dicts:
            key_sum.extend(dict_i.get(key, []))
        
        dict_merged[key] = key_sum

    return dict_merged

def get_filter_primary():
    
    filter_dict = {"highway" : "primary"}
    
    return filter_dict

def get_filter_secondary():
    
    filter_dict = {"highway" : "secondary"}

    return filter_dict

def get_filter_intercity():
    
    filter_dict = {"highway" : ["primary", "secondary"]}

    return filter_dict

def get_filter_local():
    
    filter_dict = {"highway" : ["tertiary", "living_street", "residential", "service"]}

    return filter_dict

def get_filter_cycleway():
    
    filter_cycleway = ["lane","track","opposite",
                   "shared_lane","asl","opposite_lane",
                   "separate","share_busway","sidepath",
                   "right","left","sidewalk",
                   "segregated","both","opposite_track",
                   "cyclestreet"]
    
    filter_dict = {"cycleway":filter_cycleway,
                   "highway": "cycleway"}
    filter_cycleway_tags = ["cycleway:right", "cycleway:left", "cycleway:both"]
    filter_not_no = ["yes", "seperate", "shared", "lane", "sidepath", "footway", "share_busway"]

    for tag in filter_cycleway_tags:
        filter_dict[tag] = filter_not_no

    return filter_dict

def get_filter_footway():
    filter_footway = ["sidewalk", "crossing"]
    
    filter_dict = {"highway": "footway",
                   "footway": filter_footway}
    
    return filter_dict

def get_filter_path():
    filter_path = ["sidewalk", "sidepath"]
    
    filter_dict = {"highway": "path",
                   "path" : filter_path}
    
    return filter_dict

def get_filter_bicycle():
    filter_cycleway = ["lane","track","opposite",
                       "shared_lane","asl","opposite_lane",
                       "separate","share_busway","sidepath",
                       "right","left","sidewalk",
                       "segregated","both","opposite_track",
                       "cyclestreet"]
    
    filter_bicycle = ["yes", "designated", "use_sidepath", "permissive", "official"]
    filter_cyclestreet = ["yes"]
    
    filter_dict = {
        "highway": ["cycleway", "path", "service", "track"],
        "cycleway": filter_cycleway,
        "bicycle": filter_bicycle,
        "cyclestreet": filter_cyclestreet}
    
    filter_not_no = ["yes", "seperate", "shared", "lane", "sidepath", "footway", "share_busway"]
    filter_cycleway_tags = ["cycleway:right", "cycleway:left", "cycleway:both"]
    
    for tag in filter_cycleway_tags:
        filter_dict[tag] = filter_not_no
    
    return filter_dict

def get_filter_sidewalk():
        
    filter_sidewalk_tags = ["sidewalk:right", "sidewalk:left", "sidewalk:both"]
    filter_sidewalk = ["yes", "right", "left", "seperate", "both", "shared"]
    filter_path = ["sidewalk", "sidepath"]
    filter_footway = ["sidewalk", "crossing"]
    filter_foot = ["yes", "designated", "permissive"]

    filter_dict = {
        "highway": ["pedestrian", "footway", "path", "living_street"],
        "path": filter_path,
        "footway": filter_footway,
        "sidewalk": filter_sidewalk,
        "foot": filter_foot}
    
    filter_sidewalk_tags = ["sidewalk:right", "sidewalk:left", "sidewalk:both"]
    filter_not_no = ["yes", "seperate", "shared", "lane", "sidepath", "footway", "share_busway"]

    for tag in filter_sidewalk_tags:
        filter_dict[tag] = filter_not_no
    
    return filter_dict

def get_filter_robot():
    
    filter_dict_sidewalk = get_filter_sidewalk()
    filter_dict_bicycle = get_filter_bicycle()

    merged_dict = merge_dicts_list([ filter_dict_sidewalk, filter_dict_bicycle])

    return merged_dict

def get_filter_width(streettype):
    return get_filter_tags(streettype, "width")

def get_filter_by_name(filter_name):
    filter_dict = {}
    if filter_name == "cycleway":
        filter_dict = get_filter_cycleway()
        
    if filter_name == "footway":
        filter_dict = get_filter_footway()

    if filter_name == "bicycle":
        filter_dict = get_filter_bicycle()
        
    if filter_name == "sidewalk":
        filter_dict = get_filter_sidewalk()

    if filter_name == "path":
        filter_dict = get_filter_path()

    if filter_name == "robot":
        filter_dict = get_filter_robot()
        
    if filter_name == "primary":
        filter_dict = get_filter_primary()
        
    if filter_name == "secondary":
        filter_dict = get_filter_secondary()   
        
    if filter_name == "intercity":
        filter_dict = get_filter_intercity()
        
    if filter_name == "local":
        filter_dict = get_filter_local()
            
    return filter_dict
    



if __name__ == "__main__":

    # place = "Barcelona, Spain"
    #place = "Bezirksteil Lerchenau West, München"
    # graph_paths = ["C:\\Users\\plank\\Documents\\git\\bicyclelogger\\evaluation\\src\\common"]

    # places = ["München", "Köthen", "Hoyerswerda, Sachsen", "Finsterwalde, Brandenburg"]

    # for place in places:
    #     tag_stats(place)
    ...    