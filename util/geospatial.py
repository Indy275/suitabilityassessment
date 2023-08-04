from pyproj import Transformer
import configparser
import geopandas as gpd

config = configparser.ConfigParser()
config.read('config.ini')
data_url = config['DEFAULT']['data_url']


def convert_coords(d0_loc):
    # inProj = Proj(init='epsg:4326')  # world coordinates
    # outProj = Proj(init='epsg:28992')  # Dutch coordinates
    # d0_x0, d0_x1 = transform(inProj,outProj, d0_loc[0], d0_loc[1])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992")
    # transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")
    d0_x0, d0_x1 = transformer.transform(d0_loc[1], d0_loc[0])
    return [d0_x0, d0_x1]


def get_hhnk_bbox():
    gdf = gpd.read_file(data_url + '/HHNK_gebied.shp')
    gdf = gdf.to_crs(crs="EPSG:4326")
    return ', '.join(map(str, gdf.total_bounds))