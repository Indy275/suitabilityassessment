import csv
import json
import os.path

import requests
import configparser
from joblib import dump
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from util import geospatial

import rasterio
from rasterio import features, mask

from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
json_headers = json.loads(config['DEFAULT']['json_headers'])

verbose = int(config['DATA_CREATION']['verbose'])


# def get_bbox(bbox):
#     if bbox == 'waterland':
#         waterland = gpd.read_file(data_url + "/waterland.shp")
#         bbox = [min(waterland.bounds.minx), min(waterland.bounds.miny), max(waterland.bounds.maxx),
#                 max(waterland.bounds.maxy)]
#     return bbox


def get_point_data(raster_url, point):
    point_data = requests.get(url=raster_url + "/point/", headers=json_headers, params={"geom": point}).json()
    try:
        return point_data['results'][0]['value']
    except:
        return 0


def get_raster_data(raster_url, modifier, name, bbox, width, height):
    geotiff_name_msk = data_url + '/rasters/' + modifier + '/msk_' + name + '.tiff'

    if not Path(geotiff_name_msk).is_file():
    # if True:
        print("No masked .tiff-file found for layer {}; retrieving and extracting data".format(name))
        geotiff_name = data_url + '/rasters/' + modifier + '/' + name + '.tiff'
        if not Path(geotiff_name).is_file():
        # if True:
            bbox_str = ','.join(map(str, bbox))  # API requests a string-formatted list for some reason
            r = requests.get(url=raster_url + '/data/',
                             headers=json_headers,
                             params={"bbox": bbox_str, "format": 'geotiff', "width": width, "height": height})
            with open(geotiff_name, 'wb') as f:  # Write raster data to disk
                for data in r.iter_content(None):
                    f.write(data)
        # Don't ask me how the rest of this function works; it somehow does
        gdf = gpd.read_file(data_url + '/invertedwatergang.shp')
        nodata_val = -99999
        with rasterio.open(geotiff_name) as src:  # Read raster data from disk
            raster_data = src.read()
            raster_data[raster_data < 0] = 0
            raster_meta = src.meta

        with rasterio.open(geotiff_name, 'w', **raster_meta) as dst:
            dst.write(raster_data)
        with rasterio.open(geotiff_name) as src:  # Read raster data from disk
            gdf.to_crs(crs=src.meta['crs'], inplace=True)
            masked_data, masked_transform = rasterio.mask.mask(src, gdf.geometry, nodata=nodata_val, filled=True,
                                                               all_touched=True, invert=False)
        raster_meta.update({
            'height': masked_data.shape[1],
            'width': masked_data.shape[2],
            'transform': masked_transform * src.transform,
            'crs': src.crs,
            'nodata': nodata_val
        })

        with rasterio.open(geotiff_name_msk, 'w', **raster_meta) as dst:
            dst.write(masked_data)

    else:
        print("Using found masked .tiff-file for layer {}".format(name))

    with rasterio.open(geotiff_name_msk) as f:  # Read raster data from disk
        data = f.read()

    return np.squeeze(data)


def get_labels(model, modifier, name, copy, fav_data=None, fav_name=None):
    """
    Retrieve label data from .tiff

    :param fav_data:
    :param fav_name:
    :param model: str, 'expert_ref' or 'hist_buildings'
    :param name: name of the model, equal to model
    :param copy: layer data for feature burning
    :return: labels
    """
    geotiff_name = data_url + '/rasters/' + modifier + '/' + name + '.tiff'
    if model == 'expert_ref':  # rough idea: take data points from favourite; geoms are used and value is assigned
        assert fav_data is not None and fav_name is not None
        if not Path(geotiff_name).is_file():
            print("No saved .tiff found for layer {}; retrieving data".format(name))
            json_url = data_url + '/' + fav_name + '_points.json'
            if not Path(json_url).is_file():
                with open(json_url, 'w') as f:
                    json.dump(fav_data['geometries'], f)

            with open(json_url, 'r') as f:
                points = json.load(f)

            with open(data_url + '/' + fav_name + '_pointscores.csv', 'r') as f:
                scores = [int(t[0]) - 5 for t in csv.reader(f)]

            gdf = gpd.GeoDataFrame.from_features(points, columns=['geometry', 'score'])
            gdf['value'] = np.array(scores).flatten()
            shapes = [(geom, int(value)) for geom, value in zip(gdf['geometry'], gdf['value'])]

            copy_fn = data_url + '/rasters/' + modifier + '/' + copy + '.tiff'
            copy_f = rasterio.open(copy_fn)
            meta = copy_f.meta.copy()
            meta.update(compress='lzw')
            with rasterio.open(geotiff_name, 'w+', **meta) as out:
                out_arr = out.read(1)
                burned = features.rasterize(shapes=shapes, out=out_arr, transform=out.transform, fill=0)
                out.write_band(1, burned)

    elif model == 'hist_buildings':
        if not Path(geotiff_name).is_file():
            print("No saved .tiff found for layer {}; retrieving data".format(name))

            gdf = gpd.read_file(data_url + "/waterland1900min.shp")
            gdf.to_crs(crs='EPSG:4326', inplace=True)
            gdf['value'] = 1
            shapes = ((geom, value) for geom, value in zip(gdf['geometry'], gdf['value']))

            copy_fn = data_url + '/rasters/' + modifier + '/' + copy + '.tiff'
            copy_f = rasterio.open(copy_fn)
            meta = copy_f.meta.copy()
            meta.update(compress='lzw')

            with rasterio.open(geotiff_name, 'w+', **meta) as out:
                out_arr = out.read(1)
                burned = features.rasterize(shapes=shapes, out=out_arr, transform=out.transform, fill=0)
                out.write_band(1, burned)
    else:
        print('model should be expert_ref or hist_buildings')

    with rasterio.open(geotiff_name) as f:  # Read raster data from disk
        data = f.read(1)

    # plt.imshow(np.squeeze(data))
    # plt.show()
    return np.squeeze(data)


def get_fav_data(uuid, modifier, model, bbox, width, height):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers)
    fav_name = r.json()['name']
    data = r.json()['state']

    col_names, combined_sources = [], []
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        col_names.append(layer['name'].strip())

        layerdata = np.array(get_raster_data(raster_url, modifier, layer['name'], bbox, width, height))
        combined_sources.append(layerdata)

        if verbose:
            plt.imshow(layerdata, cmap='viridis')
            plt.show()

    label_data = np.array(
        get_labels(model, modifier, name=model, copy=col_names[-2], fav_data=data, fav_name=fav_name))
    combined_sources.append(label_data)
    col_names.append(model.strip())
    return np.array(combined_sources), col_names


def create_df(fav_uuid, modifier, bbox, width, height, model):
    if not os.path.exists(data_url+'/'+modifier):
        os.makedirs(data_url+'/'+modifier)
        os.makedirs(data_url + '/rasters/' + modifier)
    extracted_data, col_names = get_fav_data(fav_uuid, modifier, model, bbox, width, height)

    extracted_data[extracted_data < -9000] = np.nan
    data_extract2 = np.transpose(extracted_data, (1, 2, 0))
    data_extract3 = np.reshape(data_extract2, (-1, data_extract2.shape[-1]))

    ss = StandardScaler()
    extracted_feats = ss.fit_transform(data_extract3[:, :-1])
    extracted_labs = data_extract3[:, -1]

    extracted = np.concatenate((extracted_feats, extracted_labs.reshape((-1, 1))), axis=1)
    df4 = pd.DataFrame(extracted, columns=col_names)
    df4.to_csv(data_url + '/' + modifier + "/" + model + ".csv", index=False)
    dump(ss, data_url + '/' + modifier + "/scaler.joblib")
    print("Data with shape {} saved to {}.csv".format(extracted.shape, modifier))
