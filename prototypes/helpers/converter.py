# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:07:51 2020

@author: plank

Functions based on:
    https://www.movable-type.co.uk/scripts/latlong.html
    https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula

    This File helps with meter <-> degree conversion.
    As well as calculating haversine distance between points.
        error margin ~0.5% because earth is not a perfect sphere.
"""

import math


def get_m_per_deg_lat(lat_mid):
    """Determine the Meters in one degree latitude at the given lat_mid.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    m_per_deg_lat = (
        111132.954
        - 559.822 * math.cos(math.radians(2.0 * lat_mid))
        + 1.175 * math.cos(math.radians(4.0 * lat_mid))
        - 0.0023 * math.cos(math.radians(6.0 * lat_mid)))
    return abs(m_per_deg_lat)


def get_m_per_deg_lon(lat_mid):
    """Determine the Meters in one degree longitude at the given lat_mid.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    m_per_deg_lon = (
        111412.84 * math.cos(math.radians(lat_mid))
        - 93.5 * math.cos(math.radians(3.0 * lat_mid))
        + 0.118 * math.cos(math.radians(5.0 * lat_mid)))

    return abs(m_per_deg_lon)


def get_deg_per_m_lon(lat_mid):
    """Determine the longitude degree value of one meter at the given lat_mid.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    return 1/get_m_per_deg_lon(lat_mid)


def get_deg_per_m_lat(lat_mid):
    """Determine the latitude degree value of one meter at the given lat_mid.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    return 1/get_m_per_deg_lat(lat_mid)


def convert_lat_to_meter(deg_lat, lat_mid):
    """Convert Degree latitude to meters at the given lat_mid.
        @arg deg_lat: The degree Latitude to convert.
        @arg lat_mid: The latitude at which to do the conversion.
    """
    return deg_lat * get_m_per_deg_lat(lat_mid)


def convert_lon_to_meter(deg_lon, lat_mid):
    """Convert Degree longitude to meters at the given lat_mid.
        @arg deg_lat: The degree longitude to convert.
        @arg lat_mid: The latitude at which to do the conversion.
    """
    return deg_lon * get_m_per_deg_lon(lat_mid)


def convert_meter_to_lat(met_lat, lat_mid):
    """Convert meters to Degree latitude at the given lat_mid.
        @arg met_lat: The amount of meters to convert.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    return met_lat * get_deg_per_m_lat(lat_mid)


def convert_meter_to_lon(met_lon, lat_mid):
    """Convert meters to Degree longitude at the given lat_mid.
        @arg met_lon: The amount of meters to convert.
        @arg lat_mid: The Latitude at which to do the conversion.
    """
    return met_lon * get_deg_per_m_lon(lat_mid)


def get_distance_haversine(lat1, lon1, lat2, lon2):
    """Calculate the haversine Distance between two points.
        @arg lat1: Latitude of Point1.
        @arg lon1: Longitude of Point1.
        @arg lat2: Latitude of Point2.
        @arg lon2: Longitude of Point2.

        @return Haversine-distance in meter
    """

    R = 6371000

    dx = math.radians(lat2 - lat1)
    dy = math.radians(lon2 - lon1)

    a = (math.sin(dx/2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        *math.sin(dy/2) **2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    d = R * c

    return d

def get_middle_point(lat1, lon1, lat2, lon2):
    """Calculate the euclidian middle-Point.
        @arg lat1: Latitude of Point1.
        @arg lon1: Longitude of Point1.
        @arg lat2: Latitude of Point2.
        @arg lon2: Longitude of Point2.

        @return center_lat, center_lon
    """

    if lat1 > lat2:
        lat_max = lat1
        lat_min = lat2
    else:
        lat_max = lat2
        lat_min = lat1

    if lon1 > lon2:
        lon_max = lon1
        lon_min = lon2
    else:
        lon_max = lon2
        lon_min = lon1

    center_lon = lon_min + ((lon_max - lon_min) / 2)
    center_lat = lat_min + ((lat_max - lat_min) / 2)

    return center_lat, center_lon


def get_middle_point_haversine(lat1, lon1, lat2, lon2):
    """Calculate the haversine middle-Point.
        @arg lat1: Latitude of Point1.
        @arg lon1: Longitude of Point1.
        @arg lat2: Latitude of Point2.
        @arg lon2: Longitude of Point2.

        @return center_lat, center_lon
    """

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    gamma1 = math.radians(lon1)
    gamma2 = math.radians(lon2)

    Bx = math.cos(phi2) * math.cos(gamma2-gamma1)
    By = math.cos(phi2) * math.sin(gamma2-gamma1)

    phi3 = math.atan2(math.sin(phi1) + math.sin(phi2),
                      math.sqrt(
                          (math.cos(phi1)+Bx)*(math.cos(phi1)+Bx) + By*By ) )
    gamma3 = gamma1 + math.atan2(By, math.cos(phi1) + Bx)

    center_lon = math.degrees(gamma3)
    center_lat = math.degrees(phi3)
    return center_lat, center_lon

def test_deg(deg, meter):
    """Testfunction for different deg_lon.
    Determines how accurate it is compared to fixed values."""
    print("--------Test Degree: {0}----------------------------".format(deg))
    m_per_deg_lon_0 = get_m_per_deg_lon(deg)
    error_margin = meter * 0.005
    error = m_per_deg_lon_0 - meter

    print_error(error, error_margin)


def test_haversine(p1, p2, act_dist):
    """Testfunction for different haversine distances.
    Determines how accurate it is compared to fixed values."""
    print("--------Test Harversine----------------------------")
    lat1, lon1 = p1
    lat2, lon2 = p2
    dist = get_distance_haversine(lat1, lon1, lat2, lon2)
    error = dist - act_dist
    error_margin = act_dist * 0.005

    print_error(error, error_margin)


def test_middle_point(p1, p2, pm):
    """Testfunction for different haversine middlepoints.
    Determines how accurate it is compared to fixed values."""
    print("--------Test Middlepoint---------------------------")
    middle_point = pm
    lat1, lon1 = p1
    lat2, lon2 = p2

    center_lat, center_lon = get_middle_point_haversine(lat1, lon1,
                                                         lat2, lon2)

    error1 = center_lat - middle_point[0]
    error_margin1 = middle_point[0] * 0.005

    error2 = center_lon - middle_point[1]
    error_margin2 = middle_point[0] * 0.005
    if ((error1 <= error_margin1)
        and (error2 <= error_margin2)):
            print_error(error1, error_margin1)
            print_error(error2, error_margin2)
    else:
        print(middle_point, center_lat, center_lon)

def print_error(error, error_margin):
    if not (error <= error_margin):
        s = "[Error] \t"
    else:
        s = "[OK]\t"
    s+= "Actual Error in m: \t\t\t{0:4f}\n\t\tacceptable Error (0.05%): \t{1:4f}\n".format(error, error_margin)

    print(s)

if __name__ == "__main__":

    """
    N/S or E/W
    at equator  E/W at 23N/S     E/W at 45N/S   E/W at 67N/S
	111.32 km	102.47 km        78.71 km        43.496 km
    """
    """
        Test for degree -> meter:
    """
    test_deg(0, 111320)
    test_deg(23, 102470)
    test_deg(45, 78710)
    test_deg(67, 43496)

    """
        Test for haversine dist.
    """
    p1 = (50.066389, -5.714722)
    p2 = (58.643889, -3.070000)
    act_dist = 968846

    test_haversine(p1, p2, act_dist)

    p1 = 50.905478, 13.329625
    p2 = 50.905128, 13.330426
    act_dist= 69

    test_haversine(p1, p2, act_dist)

    p1 = (50.066389, -5.714722)
    p2 = (58.643889, -3.070000)
    middle_point = (54.362222, -4.530556)

    test_middle_point(p1,p2,middle_point)




