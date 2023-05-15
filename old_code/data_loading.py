import pandas as pd
import geopandas as gpd
import requests
import json
import numpy as np
import shapely.ops
from pyproj import Transformer
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from rasterio.plot import show
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import dump
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio import features
from rasterio import warp
import fiona
import rioxarray
from rasterio.crs import CRS
from shapely.geometry import box
from owslib.wms import WebMapService

## DEPRECATED MESSY FILE

data_url = "C:\\Users\indy.dolmans\Documents\data"
fav_uuid = "be136f1b-ae01-4696-ad0e-f3634f4c41b6"  # , "fa3a6eb0-608b-447f-93c2-3498285f9004"  # extract active layers
# bbox = [5.02693, 52.37686, 4.69391, 52.90435]  # x1, y1, x2, y2
# bbox = [114370.516, 489022.36, 141276.49, 514152.652]  # entire NH

modifier = 'purmer'
model = 'hist_buildings'

get_poival = False
poi = 'POINT (4.942281 52.507876)'

verbose = 0  # to plot or not

if modifier == 'purmer':
    bbox = [5.04805, 52.52374, 4.93269, 52.46846]  # Purmer area
else:  # assume its schermerbeemster
    bbox = [4.78729, 52.59679, 4.88033, 52.53982]  # Schermer-Beemster area

json_headers = {
    "username": '__key__',
    "password": 'de4OeXZf.KCvG8v1AJxLzsemHXqDBMdsxPPpCmPeA',
    "Content-Type": "application/json"
}


def convert_coords(d0_loc):
    # inProj = Proj(init='epsg:4326')  # world coordinates
    # outProj = Proj(init='epsg:28992')  # Dutch coordinates
    # d0_x0, d0_x1 = transform(inProj,outProj, d0_loc[0], d0_loc[1])
    # transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992")
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")
    d0_x0, d0_x1 = transformer.transform(d0_loc[0], d0_loc[1])
    return [d0_x0, d0_x1]


def get_bbox(bbox):
    if bbox == 'waterland':
        waterland = gpd.read_file(data_url + "/waterland.shp")
        bbox = [min(waterland.bounds.minx), min(waterland.bounds.miny), max(waterland.bounds.maxx),
                max(waterland.bounds.maxy)]
    return bbox


def get_point_data(raster_url, point):
    point_data = requests.get(url=raster_url + "/point/", headers=json_headers, params={"geom": point}).json()
    try:
        return point_data['results'][0]['value']
    except:
        return 0


def get_raster_data(raster_url, modifier, name, bbox):
    geotiff_name_msk = data_url + '/rasters/' + modifier + '_' + name + '_mask.tiff'
    geotiff_name_msk = data_url + '/rasters/' + modifier + '_' + name + '.tiff'
    if not Path(geotiff_name_msk).is_file():
        print("No .tiff-file found for layer {}; retrieving and extracting data".format(name))
        bbox = get_bbox(bbox)
        w = round(np.abs(bbox[0] - bbox[2]))
        h = round(np.abs(bbox[1] - bbox[3]))

        if bbox[0] > 100000:
            bbox = convert_coords(bbox[:2]) + convert_coords(bbox[2:4])
        w = h = 400
        bbox_str = ','.join(map(str, bbox))  # API requests a string-formatted list for some reason
        r = requests.get(url=raster_url + '/data/',
                         headers=json_headers,
                         params={"bbox": bbox_str, "format": 'geotiff', "width": w, "height": h})

        geotiff_name = data_url + '/rasters/' + modifier + '_' + name + '.tiff'
        with open(geotiff_name, 'wb') as f:  # Write raster data to disk
            for data in r.iter_content(None):
                f.write(data)


        # with fiona.open(data_url + "/waterland.shp", "r") as shapefile:  # Read mask shapefile
        #     shapes = [feature["geometry"] for feature in shapefile]
        # print(shapes)
        # shapes = shapes[0]
        # mask_area = shapely.ops.unary_union(shapes)
        # with open(geotiff_name, 'rb') as f:  # Read raster data from disk
        #     out_image, out_transform = rasterio.mask.mask(f, shapes, crop=True)
        #     out_meta = out_image.meta
        # out_meta.update({"driver": "GTiff",
        #              "height": out_image.shape[1],
        #              "width": out_image.shape[2],
        #              "transform": out_transform})

        # with rasterio.open(geotiff_name_msk, "w", **out_meta) as dest:
        #     dest.write(out_image)

    else:
        print("Using found .tiff-file for layer {}".format(name))

    with open(geotiff_name_msk, 'rb') as f:  # Write raster data to disk
        data = f.read(1)

    # plt.imshow(np.squeeze(data), cmap='viridis')
    # plt.show()

    with rasterio.open(data_url + '/df.tif', 'r') as f:
        data = f.read(1)
    print(data)
    plt.imshow(np.squeeze(data), cmap='viridis')
    plt.show()

    return np.squeeze(data)


def get_labels(model, modifier, name, bbox, copy):
    """
    Retrieve label data from .tiff

    :param model: str, 'expert_ref' or 'hist_buildings'
    :param name: name of the model, equal to model
    :param bbox: bounding box
    :param copy: layer data for feature burning
    :return: labels
    """
    if model == 'expert_ref':   # rough idea: take data points from favourite; geoms are used and value is assigned
        print('to be implemented')  # TODO: implement this functionality
        # with open(data_url + '/' + fav_name + '.json') as f:
        #     geoms = json.load(f)
        # for geom in geoms:
        #     print(geom)

    elif model == 'hist_buildings':
        geotiff_name = data_url + '/rasters/' + modifier + '_' + name + '.tiff'
        if not Path(geotiff_name).is_file():
            print("No saved .tiff found for layer {}; retrieving data".format(name))
            # Redundant part since we copy bbox from other layer
            # bbox = get_bbox(bbox)
            # w = round(np.abs(bbox[0] - bbox[2]))
            # h = round(np.abs(bbox[1] - bbox[3]))
            #
            # if bbox[0] > 100000:
            #     bbox = convert_coords(bbox[:2]) + convert_coords(bbox[2:4])
            # w = h = 400

            copy_fn = data_url + '/rasters/' + modifier + '_' + copy + '.tiff'
            copy_f = rasterio.open(copy_fn)
            meta = copy_f.meta.copy()
            meta.update(compress='lzw')

            gdf = gpd.read_file(data_url + "/waterland1900min.shp")
            gdf.to_crs(crs='EPSG:4326', inplace=True)
            gdf['value'] = 1

            with rasterio.open(geotiff_name, 'w+', **meta) as out:
                out_arr = out.read(1)
                burned = features.rasterize(shapes=gdf.geometry, out=out_arr, transform=out.transform, fill=0)
                out.write_band(1, burned)

        with rasterio.open(geotiff_name) as f:  # Read raster data from disk
            data = f.read(1)

        data = data * -1  # Hacky but for now should do the trick
        data = data + 2
        return np.squeeze(data)
    else:
        print('model should be expert_ref or hist_buildings')


def get_fav_data(uuid, modifier, bbox):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers)
    fav_name = r.json()['name']
    data = r.json()['state']

    col_names, combined_sources = [], []
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        col_names.append(layer['name'].strip())

        layerdata = np.array(get_raster_data(raster_url, modifier, layer['name'], bbox))
        combined_sources.append(layerdata)

        if verbose:
            plt.imshow(layerdata, cmap='viridis')
            plt.show()

    if not Path(data_url + '/' + fav_name + '.json').is_file():
        with open(data_url + '/' + fav_name + '.json', 'w') as f:
            json.dump(data['geometries'], f)

    return combined_sources, col_names


combined_sources, col_names = get_fav_data(fav_uuid, modifier, bbox)

label_data = np.array(get_labels(model, modifier, name=model, bbox=bbox, copy=col_names[-2]))

combined_sources.append(label_data)
col_names.append(model.strip())

extracted_data = np.array(combined_sources)
extracted_data[extracted_data < -1000] = 0  # TODO fails for AHN map: water is now seen as 0NAP
data_extract2 = np.transpose(extracted_data, (1, 2, 0))
data_extract3 = np.reshape(data_extract2, (-1, data_extract2.shape[-1]))

ss = StandardScaler()
extracted_feats = ss.fit_transform(data_extract3[:, :-1])
extracted_labs = data_extract3[:, -1]

extracted = np.concatenate((extracted_feats, extracted_labs.reshape((-1, 1))), axis=1)
df4 = pd.DataFrame(extracted, columns=col_names)
df4.to_csv(data_url + '/' + modifier + ".csv", index=False)
dump(ss, data_url + '/' + modifier + "scaler.joblib")
print("Data with shape {} saved to {}.csv".format(extracted.shape, modifier))

