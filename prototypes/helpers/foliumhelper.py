# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:29:53 2020

@author: plank
"""

import os, sys
import pandas as pd
#import osmnx as ox

from pathlib import Path

import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

# from common import Foldermanager as fm
import folium
import folium.plugins
import osmnx as ox
import json
import branca
# import mplleaflet

from IPython.display import IFrame
from folium.plugins import HeatMap

# from flask import current_app


# import Osmhelper as osh


def _make_folium_polyline(
    edge, edge_color, edge_width, edge_opacity, popup_attribute=None, **kwargs
):
    """
    @Override for base Function, to alow different params, for different edges.


    Turn row GeoDataFrame into a folium PolyLine with attributes.
    Parameters
    ----------
    edge : GeoSeries
        a row from the gdf_edges GeoDataFrame
    edge_color : string
        color of the edge lines
    edge_width : numeric
        width of the edge lines
    edge_opacity : numeric
        opacity of the edge lines
    popup_attribute : string
        edge attribute to display in a pop-up when an edge is clicked, if
        None, no popup
    kwargs : dict
        Extra parameters passed through to folium
    Returns
    -------
    pl : folium.PolyLine
    """
    # check if we were able to import folium successfully
    if not folium:
        raise ImportError("The folium package must be installed to use this optional feature.")

    # locations is a list of points for the polyline
    # folium takes coords in lat,lon but geopandas provides them in lon,lat
    # so we have to flip them around
    locations = list([(lat, lng) for lng, lat in edge["geometry"].coords])

    # if popup_attribute is None, then create no pop-up
    if popup_attribute is None:
        popup = None
    else:
        # folium doesn't interpret html in the html argument (weird), so can't
        # do newlines without an iframe
        popup_text = json.dumps(edge[popup_attribute])
        popup = folium.Popup(html=popup_text)

    # create a folium polyline with attributes
    pl = folium.PolyLine(
        locations=locations,
        popup=popup,
        color=edge_color,
        weight=edge_width,
        opacity=edge_opacity,
        **kwargs,
    )
    return pl


def plot_graph_folium_colors(
    G,
    graph_map=None,
    popup_attribute=None,
    tiles="cartodbpositron",
    zoom=1,
    fit_bounds=True,
    edge_color="#333333",
    edge_width=5,
    edge_opacity=1,
    **kwargs,
):
    """
    @Override for base Function, to alow different params, for different edges.


    Plot a graph on an interactive folium web map.
    Note that anything larger than a small city can take a long time to plot
    and create a large web map file that is very slow to load as JavaScript.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    graph_map : folium.folium.Map or folium.FeatureGroup
        if not None, plot the graph on this preexisting folium map object
    popup_attribute : string
        edge attribute to display in a pop-up when an edge is clicked
    tiles : string
        name of a folium tileset
    zoom : int
        initial zoom level for the map
    fit_bounds : bool
        if True, fit the map to the boundaries of the route's edges
    edge_color : string
        color of the edge lines
    edge_width : numeric
        width of the edge lines
    edge_opacity : numeric
        opacity of the edge lines
    kwargs : dict
        Extra keyword arguments passed through to folium
    Returns
    -------
    graph_map : folium.folium.Map
    """
    # check if we were able to import folium successfully
    if not folium:
        raise ImportError("The folium package must be installed to use this optional feature.")

    # create gdf of the graph edges
    gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    # get graph centroid
    x, y = gdf_edges.unary_union.centroid.xy
    graph_centroid = (y[0], x[0])

    # create the folium web map if one wasn't passed-in
    if graph_map is None:
        graph_map = folium.Map(location=graph_centroid, zoom_start=zoom, tiles=tiles)

    multible_colors = False
    multible_widths = False


    entry_count = len(gdf_edges.index)
    if isinstance(edge_color, list):
        multible_colors = True
        if len(edge_color) < entry_count:
            raise IndexError("To few edge colors given.")
    if isinstance(edge_width, list):
        # return type(edge_width)
        multible_widths = True
        if len(edge_width) < entry_count:
            raise IndexError("To few edge widths given.")

    # add each graph edge to the map
    gdf_edges = gdf_edges.reindex()
    i = 0
    for index, row in gdf_edges.iterrows():
        
        if multible_colors:
            color = edge_color[i]

        if multible_widths:
            width = edge_width[i]
        
        i += 1

        pl = _make_folium_polyline(
            edge=row,
            edge_color=color,
            edge_width=width,
            edge_opacity=edge_opacity,
            popup_attribute=popup_attribute,
            **kwargs,
        )
        pl.add_to(graph_map)

    # if fit_bounds is True, fit the map to the bounds of the route by passing
    # list of lat-lng points as [southwest, northeast]
    if fit_bounds and isinstance(graph_map, folium.Map):
        tb = gdf_edges.total_bounds
        bounds = [(tb[1], tb[0]), (tb[3], tb[2])]
        graph_map.fit_bounds(bounds)

    return graph_map


def make_iframe_popup_edge_info(row, point):
    """Makes a HTML popup for a given Point on a Folium map."""

    max_tested = row['key']
    if len(row )< 3:
        return folium.Popup()

    html ="""
    <h1> Informations for this Point.</h1><br>
    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
      overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
      font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    </style>"""
    html += """
        <br>
            {0}
        <br>
    """.format(point)
    html += """
        <table class="tg">
            <thead>
              <tr>
                <th class="tg-0lax">{0}</th>
                <th class="tg-0lax">{1}</th>
                <th class="tg-0lax">{2}</th>
              </tr>
            </thead>
            <tbody>""".format('Tag', 'count', '%')
    #rows in the series:
    for index, value in row.items():
        html += """
        <tr>
            <th class="tg-0lax">{0}</th>
            <th class="tg-0lax">{1}</th>
            <th class="tg-0lax">{2:.2f}</th>
          </tr>
          """.format(index, value, (value/max_tested) * 100)
    html += """
    </tbody>
    </table>"""

    iframe = branca.element.IFrame(html=html, width=500, height=300)
    popup = folium.Popup(iframe, max_width=500)
    return popup

def make_iframe_popup_edge_links(edge_data, links):
    """Makes a HTML popup for a given Point on a Folium map."""

    if len(links)< 3:
        return folium.Popup()

    html ="""
    <h1> Informations for this Edge.</h1><br>
    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
      overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
      font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-0lax{text-align:left;vertical-align:top}
    </style>"""
    html += """
        <br>
            {0}
        <br>
    """.format(edge_data["edge_id"].iloc[0])
    html += """
        <table class="tg">
            <thead>
              <tr>
                <th class="tg-0lax">{0}</th>
                <th class="tg-0lax">{1}</th>
              </tr>
            </thead>
            <tbody>""".format('Key', 'Link')
    #rows in the series:
    for index, value in links.items():
        if not isinstance(value, str):
            continue
        link = Path(value)
        if link.is_absolute():
            link = link.as_uri()
            # link = link.resolve()
            # link = link.as_uri()
        key = Path(value).stem
        link_string = """<a href= "{0}" target="_blank" > {1} </a> """.format(
            link, key)

        html += """
        <tr>
            <th class="tg-0lax">{0}</th>
            <th class="tg-0lax">{1}</th>
          </tr>
          """.format(key, link_string)
    html += """
    </tbody>
    </table>"""

    #TODO add stats for the each set the edge is in.

    iframe = branca.element.IFrame(html=html, width=500, height=300)
    popup = folium.Popup(iframe, max_width=500)
    return popup


# def make_folium_map(table, plot_key = "acc_x"):
#     """Make a folium map with a vincent chart for each edge.

#     https://github.com/Leaflet/Leaflet.markercluster
#     https://python-visualization.github.io/folium/plugins.html
#     """

#     mapa = folium.Map(location=(center_lat, center_lon), zoom_start=12)

#     feature_group = folium.FeatureGroup(name='Edges')
#     options = {'disableClusteringAtZoom': 16, 'spiderfyOnMaxZoom': True}
#     marker_cluster_edges = folium.plugins.MarkerCluster(options=options)


#     for edge_id in table.edge_id.unique():
#         plot_collection = table[table['edge_id'] == edge_id]
#         number_of_rows = len(plot_collection.index)
#         multi_iter1 = {}

#         if number_of_rows == 1:
#             data = plot_collection.iloc[0]['imus']
#             if data:
#                 pdata = pd.DataFrame(data)
#                 multi_iter1[row['file']] = pdata[[plot_key]]


#         elif number_of_rows > 1:
#             longest_imu = 0
#             for row_index in range(number_of_rows):
#                 row = plot_collection.iloc[row_index]
#                 row_data = row['imus']
#                 if row_data:
#                     pdata = pd.DataFrame(row_data)
#                     pdata_length = len(pdata.index)
#                     if pdata_length > longest_imu:
#                         longest_imu = pdata_length

#                 multi_iter1[row['file']] = pdata[[plot_key]]

#             #make them all same length
#             for key in multi_iter1:
#                 tmp_data = multi_iter1[key]
#                 tmp_length = len(tmp_data)
#                 if tmp_length < longest_imu:
#                     multi_iter1[key] = fill_df_zeros(
#                         tmp_data, tmp_length, longest_imu)

#             for key in multi_iter1.keys():
#                 multi_iter1[key] = multi_iter1[key][plot_key]

#             #add index for plotting
#             index = range(0, longest_imu, 1)
#             multi_iter1['index']  = index

#         #plot
#         line = vincent.StackedArea(multi_iter1, iter_idx = 'index')
#         line.axis_titles(x='index', y='acc_x')
#         line.legend(title='acc_x über Zeit')
#         data = json.loads(line.to_json())

#         #make a foliummap with markers for each edge
#         tooltip = 'Click me!'

#         #TODO make this better
#         edge_collection = collection[collection['edge_id'].values == edge_id]

#         folium.Marker(
#             [edge_collection['latitude_match'].iloc[0],
#               edge_collection['longitude_match'].iloc[0]],
#             popup=folium.Popup(max_width=1200).add_child(
#                 folium.Vega(data, width=1200, height=700)),
#             tooltip=tooltip
#             ).add_to(marker_cluster_edges)

#     feature_group.add_child(marker_cluster_edges)
#     mapa.add_child(feature_group)
#     mapa.save('{0}.html'.format(unique_location))

# def html_link_filter(text, chars = ['\\', '\\\\', '//']):
"""https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string"""
#     for ch in text:
#         if ch in text:
#             text = text.replace(ch, "/"+ ch)

#     return text


def make_folium_map_bokeh(cfg, reset):
    all_folders = fm.get_all_folders(cfg)
    edge_folders = all_folders['dir_out_edges']

    filename_edge_links = Path(edge_folders[0], "edge_links")
    final_df = pd.read_csv(filename_edge_links)

    dirs_data_processed = all_folders['dir_out_proccessed']
    dir_data_processed = all_folders['dir_out_proccessed'][0]

    dirs_out_edges = all_folders['dir_out_edges']
    dir_out_edges = all_folders['dir_out_edges'][0]

    filename_overview = fm.load_overview(dirs_data_processed)
    if filename_overview.is_file():
        current_app.logger.info("\t Loading {0}".format(filename_overview))
        tracks = pd.read_csv(filename_overview)
    else:
        current_app.logger.error('{0} not found.'.format(filename_overview))

    for unique_location in tracks.Location.unique():
        popups = []
        coordinates = []
        filename_collection = (str(dir_data_processed)
                              + os.sep
                              + "collection_"
                              + unique_location
                              + ".csv")
        collection = pd.read_csv(filename_collection)
        edges = collection.edge_id.unique()

        for edge in edges:
            ##load edge data
            #TODO add load edge to filemanger
            filename_edge = Path(
                dir_out_edges, str(edge), str(edge)+"_imu_data")
            if not filename_edge.is_file():
                continue
            edge_data = pd.read_csv(filename_edge)
            # current_app.logger.debug(edge_data["stats"])
            #TODO How to handle multible stats from different files???
            try:
                stats = pd.DataFrame(eval(edge_data["stats"].iloc[0]))
            except NameError as e:
                current_app.logger.info(e)
                continue
            #print(edge_data.columns)
            edge_data.iloc[0].drop('imus')
            # edge_data['imu'] = eval(edge_data['imu'])

            #TODO make this better...
            xx = stats[stats.index.str.startswith('mean')]
            coord = (xx["latitude_match"].iloc[0],
                    xx["longitude_match"].iloc[0])
            coordinates.append(coord)

            links = final_df[final_df["edge_id"].values == edge]
            link = links.iloc[0].to_dict()

            popups.append(make_iframe_popup_edge_links(edge_data, link))


        #TODO center
        #point = coordinates[0]
        if coordinates:

            mapa = folium.Map(location=((50.9088, 13.3447)), zoom_start=12)

            feature_group = folium.FeatureGroup(name='Edges')
            options = {'disableClusteringAtZoom': 16, 'spiderfyOnMaxZoom': True}
            marker_cluster_edges = folium.plugins.MarkerCluster(
                coordinates, popups,options=options)

            # marker_cluster_edges = folium.plugins.MarkerCluster(coordinates, popups)
            feature_group.add_child(marker_cluster_edges)
            mapa.add_child(feature_group)
            mapa.save('{0}.html'.format(unique_location))

def map_covered_ways(city, edges, save_location = "tests"):

    #TODO get coordinates from city
    #TODO get popuplocation from edge not from gnss
    coordinates = []
    popups = []
    for edge in edges:
        coord = (edge.gnsss[0].lat_matched, edge.gnsss[0].lon_matched)
        coordinates.append(coord)
        popups.append(folium.Popup())
    mapa = folium.Map(location=((50.9088, 13.3447)), zoom_start=12)

    feature_group = folium.FeatureGroup(name='Edges')
    options = {'disableClusteringAtZoom': 16, 'spiderfyOnMaxZoom': True}
    marker_cluster_edges = folium.plugins.MarkerCluster(
        coordinates, popups,options=options)

    # marker_cluster_edges = folium.plugins.MarkerCluster(coordinates, popups)
    feature_group.add_child(marker_cluster_edges)
    mapa.add_child(feature_group)
    filename = Path(save_location, city.name + "_coveredways")
    mapa.save('{0}.html'.format(str(filename)))


def map_covered_ways_line(graph, city, used_edges, save_location = "tests"):

    nodes, edges = ox.graph_to_gdfs(graph, nodes = True, node_geometry = True,
                                    edges = True, fill_edge_geometry=True)

    color = []
    width = []

    used_ids = []

    for used_edge in used_edges:
        used_ids.append(used_edge.osm_id)

    # current_app.logger.info(edges)
    for edge in edges.itertuples():
        # current_app.logger.info(edge)
        if edge.osmid in used_ids:
            color.append("red")
            width.append("1")
        else:
            color.append(None)
            width.append("0.5")

    mapa = plot_graph_folium_colors(graph, edge_color=color, edge_width=width)

    filename = Path(save_location, city.name + "_coveredways_line")
    mapa.save('{0}.html'.format(str(filename)))

def fill_df_zeros(tmp_data, tmp_length, longest_length, columns = ['acc_x']):
    to_add = longest_length - tmp_length
    zeros = [0 for i in range(to_add)]

    # tmp_data.extend(zeros)
    tmp_frame = pd.DataFrame(zeros, columns = tmp_data.columns)

    #TODO does append need the =
    tmp_data = tmp_data.append(tmp_frame, ignore_index = True)

    return tmp_data

def get_html_link(path_to_file, link_text):
    path_to_file.replace(
        "\\\\", "\\").replace(
        "\\", "/").replace(
        "//", "/")
    link_string = """<a href= "file:///{0}" > {1} </a>""".format(
        path_to_file, link_text)

    return link_string


def highlight_edge(cfg, reset, edge_high):
    all_folders = fm.get_all_folders(cfg)
    edge_folders = all_folders['dir_out_edges']

    filename_edge_links = Path(edge_folders[0], "edge_links")
    final_df = pd.read_csv(filename_edge_links)

    dirs_data_processed = all_folders['dir_out_proccessed']
    dir_data_processed = all_folders['dir_out_proccessed'][0]

    dirs_out_edges = all_folders['dir_out_edges']
    dir_out_edges = all_folders['dir_out_edges'][0]

    filename_overview = fm.load_overview(dirs_data_processed)
    if filename_overview.is_file():
        current_app.logger.info("\t Loading {0}".format(filename_overview))
        tracks = pd.read_csv(filename_overview)
    else:
        current_app.logger.error('{0} not found.'.format(filename_overview))

    for unique_location in tracks.Location.unique():
        popups = []
        coordinates = []
        filename_collection = (str(dir_data_processed)
                              + os.sep
                              + "collection_"
                              + unique_location
                              + ".csv")
        collection = pd.read_csv(filename_collection)
        edges = collection.edge_id.unique()

        for edge in edges:
            if edge != edge_high:
                continue
            ##load edge data
            #TODO add load edge to filemanger
            filename_edge = Path(
                dir_out_edges, str(edge), str(edge)+"_imu_data")
            if not filename_edge.is_file():
                continue
            edge_data = pd.read_csv(filename_edge)
            # current_app.logger.info(edge_data["stats"])
            #TODO How to handle multible stats from different files???
            try:
                stats = pd.DataFrame(eval(edge_data["stats"].iloc[0]))
            except NameError as e:
                current_app.logger.error(e)
                continue
            #print(edge_data.columns)
            edge_data.iloc[0].drop('imus')
            # edge_data['imu'] = eval(edge_data['imu'])

            #TODO make this better...
            xx = stats[stats.index.str.startswith('mean')]
            coord = (xx["latitude_match"].iloc[0],
                    xx["longitude_match"].iloc[0])
            coordinates.append(coord)

            links = final_df[final_df["edge_id"].values == edge]
            link = links.iloc[0].to_dict()

            popups.append(make_iframe_popup_edge_links(edge_data, link))


        #TODO center
        #point = coordinates[0]
        if coordinates:

            mapa = folium.Map(location=((50.9088, 13.3447)), zoom_start=12)

            feature_group = folium.FeatureGroup(name='Edges')
            options = {'disableClusteringAtZoom': 16, 'spiderfyOnMaxZoom': True}
            marker_cluster_edges = folium.plugins.MarkerCluster(
                coordinates, popups,options=options)

            # marker_cluster_edges = folium.plugins.MarkerCluster(coordinates, popups)
            feature_group.add_child(marker_cluster_edges)
            mapa.add_child(feature_group)
            mapa.save('{0}.html'.format(edge_high))


def map_match(graph, df_gnss, save_location = "tests"):

    folium_map = ox.plot_graph_folium(
        graph, popup_attribute='osmid', edge_width=2)


    for row in df_gnss.itertuples():
        folium.Circle((row.lat, row.lon),
                radius = 1, color = "grey").add_to(folium_map)

        if row.user_tracktype == 0:
            row_tupel = (row.lat_matched, row.lon_matched)
            row_tooltip = "{}, {}".format(str(row.osm_id), str(row.user_tracktype))
            row_radius = 1.5
            row_color = 'blue'
        elif row.user_tracktype in row.determined_street_types:
            row_tupel = (row.lat_matched, row.lon_matched)
            row_tooltip = "{0} , {1} , {2}".format(
                str(row.osm_id), str(row.user_tracktype),
                str(row.determined_street_types))
            row_radius = 1.5
            row_color = 'green'
        else:
            row_tupel = (row.lat_matched, row.lon_matched)
            row_tooltip = "{0} , {1} , {2}".format(
                str(row.osm_id), str(row.user_tracktype),
                str(row.determined_street_types))
            row_radius = 1.5
            row_color = 'red'

        #print(row)
        popup = make_iframe_popup(row)
        folium.Circle(row_tupel, tooltip=row_tooltip, popup = popup,
                      radius=row_radius, color=row_color).add_to(folium_map)

    path_file_mm_html = Path(save_location, "MapMatching.html")
    #print(path_file_mm_html)
    folium_map.save(str(path_file_mm_html))
    IFrame(path_file_mm_html, width=1000, height=500)


def make_iframe_popup(row):
    """Makes a fancy HTML popup for a given Point on a Folium map."""
    #TODO highlight big acc
    #TODO make a plot of the acc

    if len(row )> 3:
        # current_app.logger.info(row)
        # current_app.logger.info(row.values)
        # for index, value in row.items():
        #     current_app.logger.info("Index : {0}, Value : {1}".format(index, value))

        row_dict = {"Name" : row.name,
                 "osm_id": row.osm_id,
                 "lat": row.lat,
                 "lon": row.lon,
                 "lat_matched" : row.lat_matched,
                 "lon_matched" : row.lon_matched,
                 "speed": row.speed,
                 "course": row.course,
                 "hdop" : row.hdop,
                 "sat_count" : row.number_of_sat,
                 "altitude" :row.altitude,
                 "dist_v" : row.dist_v
                 }

        html ="""
        <h1> Informations for this Point.</h1><br>
        <style type="text/css">
        .tg  {border-collapse:collapse;border-spacing:0;}
        .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
          overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
          font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
        .tg .tg-0lax{text-align:left;vertical-align:top}
        </style>"""
        #rows in the series:
        for index, value in row_dict.items():
            if index == 'Name':
                html += """
                <table class="tg">
                <thead>
                  <tr>
                    <th class="tg-0lax">{0}</th>
                    <th class="tg-0lax">{1}</th>
                  </tr>
                </thead>
                <tbody>""".format(index, value)
            else:
                html += """
                <tr>
                    <th class="tg-0lax">{0}</th>
                    <th class="tg-0lax">{1}</th>
                  </tr>
                  """.format(index, value)
        html += """
        </tbody>
        </table>"""
    else:

        html ="""
        <h1> Informations for this Point.</h1><br>
        {0}
        """.format(row)

    #print(html)

    iframe = branca.element.IFrame(html=html, width=500, height=300)
    popup = folium.Popup(iframe, max_width=500)
    return popup

def map_acc(graph, df_gnss, save_location, filename):

    folium_map = ox.plot_graph_folium(
        graph, popup_attribute='osmid', edge_width=2)

    HeatMap(data=df_gnss[['lat', 'lon', 'speed']].groupby(
        ['lat', 'lon']).sum().reset_index().values.tolist(),
        radius=8, max_zoom=13).add_to(folium_map)

    path_file_mm_html = Path(save_location, "Speed_Heatmap.html")
    folium_map.save(str(path_file_mm_html))
    IFrame(path_file_mm_html, width=1000, height=500)


def folium_barcelona(city_name = "München, Bayern"):

    #graph = ox.load_graphml("Barcelona.ml")
    graph_path = r"C:\Users\plank\Documents\git\bicyclelogger\data\in\osm"
    graph_paths = []
    graph_paths.append(graph_path)
    graph = osh.get_place(city_name, graph_paths = graph_paths, simplify = True, flask=False)

    edges = ox.graph_to_gdfs(graph, nodes = False, edges = True, fill_edge_geometry=True)

    #print(edges)
    colors = []
    widths = []
    counter = 0
    # current_app.logger.info(edges.iloc[-1]["osmid"])

    # tags = []
    # for edge in edges.itertuples():
    #     if edge.highway not in tags:
    #         tags.append(edge.highway)

    # current_app.logger.info(tags)

    # filter_tags = []

    # input()

    for edge in edges.itertuples():
        # current_app.logger.info(edge)
        # current_app.logger.info(edge)
        if edge.highway == "street":
            colors.append(None)
            widths.append(1)
        else:

            counter += 1
            if counter % 1000 == 0 :
                pass
                # current_app.logger.info("{}".format(counter))
                # current_app.logger.info(edge)
            try:
                if not pd.isnull(edge.width):
                    colors.append("green")
                    widths.append(1)
                else:
                    # current_app.logger.debug("jup")
                    colors.append("grey")
                    widths.append(0.1)
            except Exception as e:
                # current_app.logger.error(e)
                # current_app.logger.error("{}".format(edge.width))
                colors.append("grey")
                widths.append(1)


    # current_app.logger.info("Saving graph...")
    # current_app.logger.debug("{}".format(len(colors)))
    # current_app.logger.debug("{}".format(len(widths)))
    folium_map = plot_graph_folium_colors(graph,
                                      edge_width = widths, edge_color = colors)
    path_file_mm_html = Path("{}.html".format(city_name))
    folium_map.save(str(path_file_mm_html))
    IFrame(path_file_mm_html, width=1000, height = 500)


if __name__ == "__main__":
    cfg = fm.get_config_cmd()

    #make_folium_map_bokeh(cfg)

    #highlight_edge(cfg, False, 517)

    folium_barcelona("Amsterdam")