import geopandas as gpd
import fiona
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html plotting of gpd
data_url = "C:\\Users\indy.dolmans\Documents\data"


def create_polygon(p, size):
    h_size = size / 2
    return Polygon([[p.x - h_size, p.y - h_size],
                    [p.x + h_size, p.y - h_size],
                    [p.x + h_size, p.y + h_size],
                    [p.x - h_size, p.y + h_size]]
                   )


def average_values(data, p, resolution=1):
    poly_size = int(np.sqrt(p.area))  # area = h times w so the line size is the sqrt for h==w
    poly_size = np.round(poly_size)
    values, c = 0, 0
    xx, yy = p.exterior.coords.xy
    for i in range(0, poly_size, resolution):
        for j in range(0, poly_size, resolution):
            x = xx[0] + i
            y = yy[0] + j
            values += [d[0] for d in data.sample([(x, y)])][0]
            # TODO handle non-data points
            c += 1
    return values / c


def poly_val_list(data, coord_list):
    avg_per_poly = []
    for p in coord_list:
        poly = create_polygon(p, 100)
        avg_per_poly.append(average_values(data, poly))
    return avg_per_poly


frame = gpd.read_file(data_url + '\cbsgebiedsindelingen2022.gpkg')
print(frame.columns)
print(frame.head(15))

frame = gpd.read_file(data_url + '\BestuurlijkeGebieden_2023.gpkg')
print(frame.columns)
print(frame.iloc[1])

points = [Point(130000, 521000), Point(127000, 515000), Point(145000, 526000), Point(125000, 535000)]
polygons = [create_polygon(p, 20) for p in points]
gdf = gpd.GeoDataFrame([1, 2, 3, 4], geometry=points, crs=28992)
gdf_polygon = gpd.GeoDataFrame([1, 2, 3, 4], geometry=polygons, crs=28992)
coord_list = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]

suitability_score = [9, 5, 7, 3]
gdf['s_score'] = suitability_score

fig, ax = plt.subplots()
with rasterio.open(data_url + "\Overstroming.tif") as src:
    extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    print(extent)
    print(points)
    data = src.read(1)
    # rasterio.plot.show(data, ax=ax)
    ax.imshow(data, cmap='terrain', vmin=min(np.ravel(data)), vmax=0)
    # gdf.plot(ax=ax, missing_kwds=dict(color="lightgrey", ))
    # gdf_polygon.plot(ax=ax)
    plt.show()
    gdf['Overstroming'] = [x[0] for x in src.sample(coord_list)]
    gdf['Overstravg'] = poly_val_list(src, points)

with rasterio.open(data_url + "\draagkracht.tif") as src:
    gdf['Draagkracht'] = [x[0] for x in src.sample(coord_list)]
    gdf['Draagkravg'] = poly_val_list(src, points)

print(gdf.head())
print(gdf.dtypes)
print(gdf.describe(include='all'))

print(gdf.columns.tolist())
gdf.drop(0, axis=1, inplace=True)
gdf.to_file(filename=data_url + '\BodemWater1.shp.zip', driver='ESRI Shapefile')
gdf = gpd.read_file(data_url + '\BodemWater1.shp.zip')
