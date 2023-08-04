# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:43:06 2021

@author: plank
"""

from configobj import ConfigObj

def get_cities(cfg):
    """Returns list of cities."""
    return multi_line_to_array(cfg["Cities"]["cities"])

def get_colors(cfg):
    """Returns list of colors."""
    return multi_line_to_array(cfg["Plot"]["colors"])

def get_scenario_names(cfg):
    """Returns the names of all scenarios."""
    szs = []
    for sz in cfg["Scenarios"]:
        szs.append(cfg["Scenarios"][sz]["name"])

    return szs

def get_markers(cfg):
    """Returns all markers."""
    return multi_line_to_array(cfg["Plot"]["markers"])

def get_extra_markers(cfg):
    """Returns the extra markers."""
    unicode_numbers = multi_line_to_array(cfg["Plot"]["extra_markers"])
    return unicode_numbers

def get_override_filter(cfg, override_name):
    """Returns dict with all override values."""
    d = {}
    for key in cfg["Tags"]["Overrides"][override_name]:
        d[key] = multi_line_to_array(cfg["Tags"]["Overrides"][override_name][key])
    return d

def yield_scenarios(cfg):
    """Returns generator with all scenario names."""
    for sz in cfg["Scenarios"]:
        yield cfg["Scenarios"][sz]

def get_simplify(cfg):
    """Returns cfg["Parameter"].as_bool("simplify")"""
    return cfg["Parameter"].as_bool("simplify")

def get_speeds(cfg, scenario):
    """Builds dictionary with allowed speeds per filter_name.

    Parameters
    ----------
    cfg : ConfigObj
        ConfigObj
    scenario : str
        name of the scenario for which to build the dict.

    Returns
    -------
    speeds : dict
        dictionary with filter_name:speed.

    Example
    -------
    {'intercity': 0, 'local': 0, 'bicycle': 0, 'sidewalk': 5,
    'overrides': {'living_street': 5}}
    """
    d = {}
    for speed in cfg["Scenarios"][scenario]["Speeds"]:
        d[speed] = cfg["Scenarios"][scenario]["Speeds"].as_int(speed)
        
    d["overrides"] = {}
    for detail in cfg["Scenarios"][scenario]["Details"]:
        d["overrides"][detail] = cfg["Scenarios"][scenario]["Details"].as_int(detail)
                
    return d

def get_scenarios(cfg):
    """Returns list of scenario names."""
    l = []
    for sz in cfg["Scenarios"]:
        l.append(cfg["Scenarios"][sz])
        
    return l

def get_short_names(cfg):
    """Returns list of short scenario names."""
    l = []
    for sz in cfg["Scenarios"]:
        l.append(cfg["Scenarios"][sz]["short"])
        
    return l

def get_traveltimes(cfg):
    """Returns list of traveltimes"""
    l = multi_line_to_array(cfg["Parameter"]["traveltimes"])
    ln = []
    for e in l:
        ln.append(int(e))
    return ln

def get_filter_params(cfg, filter_name):
    """Gets all osm tags associated with a filter.

    Parameters
    ----------
    cfg : ConfigObj
        The Config
    filter_name : str
        name of the filter

    Returns
    -------
    filter : dict
        dictionary with all valid osm key value pairs associated with a filter.

    Example
    -------
    > get_filter_params(cfg, "Intercity")
    > {'highway': ['primary', 'secondary']}
    """
    d = {}
    for key in cfg["Tags"][filter_name]:
        if key == "extra_tags":
            for entry in multi_line_to_array(
                            cfg["Tags"][filter_name]["extra_tags"]):
                d[entry] = multi_line_to_array(
                                cfg["Tags"]["NotNo"]["notno"])
        else:
            d[key] = multi_line_to_array(cfg["Tags"][filter_name][key])
        
    return d

def get_filter_names(cfg):
    """Returns the filter names."""
    return [x for x in cfg["Tags"] if 
            x != "NotNo" 
            and x != "UsefulTagsWay"
            and x != "Overrides"]

def get_filters(cfg):
    """Returns get_filter_params for each filter_name"""
    filters = {}    
    for filter_name in get_filter_names(cfg):
        filters[filter_name] = get_filter_params(cfg, filter_name)  
        
    return filters

def get_relevant_tags(cfg):
    """Returns a list of all tags that could be used."""
    s = set()
    for filter_name in get_filter_names(cfg):
        d = get_filter_params(cfg, filter_name)
        for key, value in d.items():
            s.add(key)
    
    u = set(multi_line_to_array(cfg["Tags"]["UsefulTagsWay"]["usefultagsway"]))
    return list(s | u)

def get_which_results(cfg):
    stringarray = multi_line_to_array(cfg["Cities"]["which_results"])
    return [int(x) for x in stringarray]

def multi_line_to_array(ml):
    """Returns a list of a multiline string."""
    mla = ml.split("\n")[1:-1]
    cities = []
    for entry in mla:
        cities.append(entry.lstrip())
    return cities
    # print(cities)

def get_override(cfg):
    """Returns cfg["Parameter"].as_bool("override")""" 
    override = False 
    try:
        override = cfg["Parameter"].as_bool("override")
    except:
        override = False 
    return override

if __name__ == "__main__":
    
    cfg = ConfigObj("ready4robots.ini")
    
    # print(get_speeds(cfg, "Small"))
    print(get_filter_params(cfg, "Intercity"))
    # print(get_filter_params(cfg, "Medium"))
    # print(get_relevant_tags(cfg))
    # print(get_extra_markers(cfg))
    # print(get_override(cfg))