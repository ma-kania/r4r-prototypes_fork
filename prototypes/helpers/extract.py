# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:44:28 2020

@author: plank
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler, SMTPHandler
from pathlib import Path

import geopandas as geopd
import numpy as np
import osmnx as ox
import pandas as pd

from . import converter as cv
from . import osmhelper as oh

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


def extract_place(place="Weißenborn, Sachsen",
                  file_input=r"csv_Bevoelkerung_100m_Gitter/lat-lon-deci-Einw.csv",
                  lat1=0, lon1=0, lat2=0, lon2=0,
                  override=False, bbox = None, which_result=0):
    """[summary]

    Parameters
    ----------
    place : str, optional
        [description], by default "Weißenborn, Sachsen"
    file_input : regexp, optional
        [description], by default r"csv_Bevoelkerung_100m_Gitter/lat-lon-deci-Einw.csv"
    lat1 : int, optional
        [description], by default 0
    lon1 : int, optional
        [description], by default 0
    lat2 : int, optional
        [description], by default 0
    lon2 : int, optional
        [description], by default 0
    override : bool, optional
        [description], by default False
    """

    file100 = Path("csvs", "{}.csv".format(place))

    if Path(file100).is_file() and not override:
        return

    if [lat1, lon1, lat2, lon2].count(0) == 4:
        bbox = ox.geocoder.geocode_to_gdf(place, buffer_dist=200,which_result=which_result)
        lat1 = bbox.bbox_north[0]
        lat2 = bbox.bbox_south[0]
        lon1 = bbox.bbox_west[0]
        lon2 = bbox.bbox_east[0]
        print(bbox)

    df = pd.read_csv(file_input)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"},
              inplace=True)

    extracted = df[
        (df["lat"] > lat2) &
        (df["lat"] < lat1) &
        (df["lon"] > lon1) &
        (df["lon"] < lon2)]

    test_geom2 = bbox.geometry.loc[0]

    gdf = geopd.GeoDataFrame(
        extracted, geometry=geopd.points_from_xy(extracted.lon, extracted.lat))

    gdf2 = gdf.loc[gdf.within(test_geom2)]

    city_cut = pd.DataFrame(gdf2.drop(columns='geometry'))
    city_cut = city_cut[city_cut["einwohner"] > 0]
    # print(city_cut.shape)

    city_cut.to_csv(file100)
    print("wrote {}".format(file100))

def extract_place_easy(file_100 = "",
                  file_input=r"csv_Bevoelkerung_100m_Gitter/lat-lon-deci-Einw.csv",
                  lat1=0, lon1=0, lat2=0, lon2=0,
                  override=False, bbox = None):
    """[summary]

    Parameters
    ----------
    place : str, optional
        [description], by default "Weißenborn, Sachsen"
    file_input : regexp, optional
        [description], by default r"csv_Bevoelkerung_100m_Gitter/lat-lon-deci-Einw.csv"
    lat1 : int, optional
        [description], by default 0
    lon1 : int, optional
        [description], by default 0
    lat2 : int, optional
        [description], by default 0
    lon2 : int, optional
        [description], by default 0
    override : bool, optional
        [description], by default False
    """

    if Path(file_100).is_file() and not override:
        return

    df = pd.read_csv(file_input)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"},
              inplace=True)

    extracted = df[
        (df["lat"] > lat2) &
        (df["lat"] < lat1) &
        (df["lon"] > lon1) &
        (df["lon"] < lon2)]

    # test_geom2 = bbox.geometry.loc[0]

    gdf = geopd.GeoDataFrame(
        extracted, geometry=geopd.points_from_xy(extracted.lon, extracted.lat))

    city_cut = pd.DataFrame(gdf.drop(columns='geometry'))
    city_cut = city_cut[city_cut["einwohner"] > 0]

    city_cut.to_csv(file_100)
    print("wrote {}".format(file_100))

def extract_place_easy_multiple(files = [],
                  file_input=r"csv_Bevoelkerung_100m_Gitter/lat-lon-deci-Einw.csv",
                  lat1=[], lon1=[], lat2=[], lon2=[],
                  override=False, bbox = None):

    need_to_load = False
    for i, file in enumerate(files):
        if Path(file).is_file() and not override:
            continue
        need_to_load = True

    if not need_to_load: return

    df = pd.read_csv(file_input)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"},
              inplace=True)
    for i, file in enumerate(files):
        if Path(file).is_file() and not override:
            continue
    
        print(str(file))
        print(lat1[i], lat2[i], lon1[i], lon2[i])
        extracted = df[
            (df["lat"] > lat2[i]) &
            (df["lat"] < lat1[i]) &
            (df["lon"] > lon1[i]) &
            (df["lon"] < lon2[i])]

    # test_geom2 = bbox.geometry.loc[0]

        gdf = geopd.GeoDataFrame(
            extracted, geometry=geopd.points_from_xy(extracted.lon, extracted.lat))

        city_cut = pd.DataFrame(gdf.drop(columns='geometry'))
        city_cut = city_cut[city_cut["einwohner"] > 0]
        city_cut.to_csv(file)
        print("wrote {}".format(file))

def change_resolution(df, old_length, new_length):
    """Evenly distributes the population in a square 
    of length n in square(s) of size m.

    Parameters
    ----------
    df : DataFrame
        Dataframe with lat, lon of the middlepoint of the old squares 
        as well as the population.
    old_length : int
        edge length of the original square
    new_length : [type]
        edge length of the new square(s)

    Returns
    -------
    DataFrame
        Lat, Lon of the middlepoint of the new squares 
        as well as the population
    """
    list_x = []
    list_y = []
    list_z = []

    # df.rename( columns = {"0": "lon", "1" : "lat", "2": "einwohner"}, inplace=True)

    resolution_factor = int(old_length / new_length)

    lat = df.iloc[0]["lat"]

    #100 is standard
    m_per_lon = cv.get_deg_per_m_lon(lat) * new_length
    m_per_lat = cv.get_deg_per_m_lat(lat) * new_length

    res_half = int(resolution_factor/2)

    for row in df.itertuples():
        e = row.einwohner / resolution_factor**2
        for i in range(-res_half, res_half):
            for j in range(-res_half, res_half):
                list_x.append(row.lat + i * m_per_lat)
                list_y.append(row.lon + j * m_per_lon)
                list_z.append(e)

    return pd.DataFrame(list(zip(list_y, list_x, list_z)))


def change_resolution_place(place="Freiberg, Sachsen", old=100, new=10, override=False):
    """Generates a csv File with new population resolution."""
    file100 = r"csvs/{}.csv".format(place)
    file10 = r"csvs/{}-10m.csv".format(place)

    if Path(file10).is_file() and not override:
        return

    df = pd.read_csv(file100)

    converted = change_resolution(df, old_length=old, new_length=new)

    converted.to_csv(file10)


def get_pop_total(place):
    """Returns the total population in the graph."""
    gridcsv = r"csvs/{}-10m.csv".format(place)
    df = pd.read_csv(gridcsv)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"}, inplace=True)
    pop = df["einwohner"].sum()

    return pop

def get_pop_total_100(gridcsv):
    """Returns the total population in the graph."""
    # gridcsv = r"csvs/{}-100m.csv".format(place)
    df = pd.read_csv(gridcsv)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"}, inplace=True)
    pop = df["einwohner"].sum()

    return pop

def test():
    return "ok"

def get_pop_sum_per_node(lat, lon, pops, 
                         current_node, closest_middle_points, max_dist=85):
    """Sums population data per osmid 
    for each valid reachable square middlepoint.

    Parameters
    ----------
    lat : Series
        lat of all nodes
    lon : Series
        lon of all nodes
    pops : Series
        pop of all nodes
    current_node : int
        osm_id of the node for which to sum
    closest_middle_points : list
        all squares that resolve this node as closest.
    max_dist : int, optional
        maximum distance for assigning. 
        If distance between middlepoint and node is bigger 
        population is not assigned, by default 85

    Returns
    -------
    float
        sum of all reachable population
    """
    pop_sum = 0

    for middle_point in closest_middle_points:
        dist = cv.get_distance_haversine(
            current_node["y"], current_node["x"],
            lat[middle_point], lon[middle_point])
        # 85m is about 1min of walking.
        if dist < max_dist:
            pop = pops.iloc[middle_point]
            if pop > 0:
                pop_sum += pop
    if pop_sum == 0:
        pop_sum = -1
    return pop_sum


def assign_pop(graph, city_csv, grid_pop_file):
    """Assign population data to each node in the graph.

    Parameters
    ----------
    graph : graph
        Graph to which to assign to.
    city_csv : str
        Filename for the cities population data.
    grid_pop_file : str
        Filename to save to.

    Returns
    -------
    Nodes : DataFrame
        ['highway', 'street_count', 'pop']
    """
    df = pd.read_csv(city_csv)
    df.rename(columns={"0": "lon", "1": "lat", "2": "einwohner"}, inplace=True)

    x = df["lat"]
    y = df["lon"]
    z = df["einwohner"]
    max_pop = z.sum()
    print("Max pop: {}".format(max_pop))
    nearest_nodes_to_middlepoints = ox.get_nearest_nodes(graph, y, x, method="balltree")

    nodes = ox.graph_to_gdfs(graph, edges = False)

    list_pop = []
    for node in nodes.itertuples():
        closest_middle_points, = np.where(nearest_nodes_to_middlepoints == node.Index)
        if closest_middle_points.any():
            pop_sum = get_pop_sum_per_node(
                x, y, z, graph.nodes[node.Index], closest_middle_points)
            pop_sum = round(pop_sum, 4)
        else:
            pop_sum = -1
        list_pop.append(pop_sum)

    nodes["einwohner"] = list_pop
    nodes = nodes.drop(columns=["x", "y", "geometry"])
    copy = nodes.drop(columns=["highway"]) # drop non numeric for sum
    c_pop = copy.sum()[1]
    print(c_pop)
    print("filtered: {}".format(int((c_pop / max_pop)*100)))
    print(nodes.columns)
    nodes.to_csv(grid_pop_file)
    df = pd.read_csv(str(grid_pop_file))
    return df


def get_pop_per_osmid(place="Freiberg, Sachsen",
                      gridcsv=None,
                      graph=None,
                      osm_ins=None,
                      logger=None,
                      override=False,
                      simplify=True,
                      file_identifier="",
                      which_result=0):
    """Get population data for each node in graph.

    Parameters
    ----------
    place : str, optional
        [description], by default "Freiberg, Sachsen"
    gridcsv : [type], optional
        [description], by default None
    graph : [type], optional
        [description], by default None
    osm_ins : [type], optional
        [description], by default None
    logger : [type], optional
        [description], by default None
    override : bool, optional
        [description], by default False
    simplify : bool, optional
        [description], by default True
    file_identifier : str, optional
        [description], by default ""

    Returns
    -------
    [type]
        [description]
    """

    simple = "_simple" if simplify else "_complex"
    
    if logger is None:
        logger = get_logger()

    if gridcsv is None:
        gridcsv = r"csvs/{}-10m.csv".format(place)

    if osm_ins is None:
        osm_in = "osm"
        osm_ins = [osm_in]
    if (file_identifier is None) or (file_identifier == ""):
        grid_pop_file = Path("csvs", place + "_gridpop" + simple + ".csv")
    else:
        grid_pop_file = Path("csvs", place + "_" +
                             file_identifier + "_gridpop" + simple + ".csv")

    if grid_pop_file.is_file() and not override:
        df = pd.read_csv(str(grid_pop_file))
        return df

    if graph is None:
        graph = oh.get_place(place, osm_ins, simplify=simplify,
                             logger=logger, force_download=False,
                             file_identifier=file_identifier,
                             which_result=which_result)

    return assign_pop(graph, gridcsv, grid_pop_file)


def get_logger():
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/paper.log',
                                       maxBytes=131072, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s]: %(message)s '
        '[in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.DEBUG)

    # extract_Freiberg()
    # change_res_Freiberg()
    # heat_map()
    # nearest_Node_assign()
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    return logger

def get_population_data(
        place="Freiberg, Sachsen", file_identifier="",
        simplify=True, override=False,
        useful_tags_way=None, which_result=0):
    """[summary]

    Parameters
    ----------
    place : str, optional
        [description], by default "Freiberg, Sachsen"
    file_identifier : str, optional
        [description], by default ""
    simplify : bool, optional
        [description], by default True
    override : bool, optional
        [description], by default False
    useful_tags_way : [type], optional
        [description], by default None
    """
    # TODO instead of place name pass graph so it doesnt have to be loaded each time...
    if not useful_tags_way is None:
        ox.config(useful_tags_way=useful_tags_way)
    print("Extracting...")
    extract_place(place, override=override, which_result=which_result)
    print("Changing Resolution")
    change_resolution_place(place, override=override)
    print("Linking Pop to OSMID...")
    get_pop_per_osmid(place, simplify=simplify,
                      file_identifier="", override=override,
                      which_result=which_result)

    print("Done")


if __name__ == '__main__':

    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/paper.log',
                                       maxBytes=131072, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] [%(levelname)s]: %(message)s '
        '[in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.DEBUG)

    place = "Weißenborn, Sachsen"
    place = "Freiberg, Sachsen"
    place = "Schkeuditz"
    get_population_data(place=place, file_identifier="",
                        simplify=True, override=True)
