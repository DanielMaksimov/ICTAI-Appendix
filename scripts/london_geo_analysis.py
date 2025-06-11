import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point, shape, mapping
from sklearn.cluster import KMeans
import pickle
from geopy.geocoders import Nominatim
import fiona
from shapely.ops import unary_union
from collections import Counter
from functools import reduce
import operator
from collections import Counter

# ********** This script contains the following functions: **********
# voronoi_construction: code for getting the voronoi map along with cluster labels
# Other utilitary functions whose aim is explained in the function's documentation
# *******************************************************************

def voronoi_construction():
  # Bounding box around London's boroughs
  limit = Polygon([
              (-0.524, 51.71),
              (0.373, 51.71),
              (0.373, 51.27),
              (-0.524, 51.27)
          ])
  
  user_coord = []
  coordinates_list = []
  for tweet in tw_l:
      tweet_coordinates = tweet['geo']['coordinates']['coordinates']
      point = Point(tweet_coordinates[0], tweet_coordinates[1])
      if limit.contains(point):
          user_coord.append({"user":tweet['author_id'], "coord":tweet_coordinates})
          coordinates_list.append(tweet_coordinates)
  coordinates_list = np.array(coordinates_list)
  clustering = KMeans(n_clusters=10, n_init='auto').fit_predict(coordinates_list)
  
  # CREATING BUFFER AROUND BOUND FOR CLEAN VORONOI
  bound = limit.buffer(100).envelope.boundary
  boundarypoints = [bound.interpolate(distance=d) for d in range(0, np.ceil(bound.length).astype(int), 100)]
  boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])
  creation_list = np.concatenate((coordinates_list, boundarycoords))
  
  # VORONOI
  vor = Voronoi(creation_list)
  lines = [shapely.geometry.LineString(vor.vertices[line]) for line in
           vor.ridge_vertices if -1 not in line]
  
  polys = shapely.ops.polygonize(lines)
  voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs="epsg:4326")
  
  polydf = gpd.GeoDataFrame(geometry=[limit], crs="epsg:4326")
  points = gpd.GeoDataFrame(
      geometry=gpd.points_from_xy(x=coordinates_list[:, 0], y=coordinates_list[:, 1], crs="epsg:4326"))
  
  result = gpd.overlay(df1=voronois, df2=polydf, how="intersection") # First result

  group_polygons = {groupe: [] for groupe in set(clustering)}

  # Assigning each polygon of our first result its cluster label
  for index_list, point_list in enumerate(coordinates_list):
      search_point = Point(point_list[0], point_list[1])
      that_polygon = result[result.geometry.contains(search_point)]
      groupe = int(clustering[index_list])
      group_polygons[groupe].append(that_polygon)
  
  gdf_list = []
  for groupe, polygons in group_polygons.items():
      merged_polygon = unary_union(polygons)
      group_polygons[groupe] = merged_polygon.buffer(0) #Fills the small holes in the polygon 
  
  for groupe, polygon in group_polygons.items():
      gdf = gpd.GeoDataFrame({'Groupe': [groupe]}, geometry=[polygon])
      gdf_list.append(gdf)
  
  # Concatenation of all GeoDataFrames to one
  gdf_all = pd.concat(gdf_list, ignore_index=True)
  gdf_all.crs = 'EPSG:4326'
  gdf_all.to_file("my_gdf.shp")

# Sometimes the resulting .shp may contain small holes, use the following function in order to fill them
  def fill_holes(polygon):
      if polygon.geom_type == 'Polygon':
          return Polygon(polygon.exterior)
      elif polygon.geom_type == 'MultiPolygon':
          return MultiPolygon([Polygon(geom.exterior) for geom in polygon])
      return polygon

  gdf_all['geometry'] = gdf_all['geometry'].apply(fill_holes)
  gdf_all.to_file("final_voronoi_result.shp")


def get_shapefile_from_cluster(coordinates_list, clustering, path):
    """
    Returns a shapefile of points with associated cluster labels
      :param coordinates_list: a list of the coordinates of each point
      :param clustering: a list of each point's corresponding cluster label
      :param path: str, the path to the output shapefile
    """
    loc = Nominatim(user_agent="GetLoc", timeout=10)
    schema = {
        'geometry': 'Point',
        'properties': [('Group', 'str')]
    }
    pointshp = fiona.open(f'{path}', mode='w',
                          driver='ESRI Shapefile',
                          schema=schema,
                          crs="EPSG:4326")

    for pos, point in enumerate(coordinates_list):
        rowdict = {
            'geometry': {'type': 'Point',
                         'coordinates': (point[0], point[1])},
            'properties': {'Group': int(clustering[pos])}
        }
        pointshp.write(rowdict)
    pointshp.close()
    return 0
