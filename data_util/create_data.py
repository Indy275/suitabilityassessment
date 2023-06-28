import json
import os.path
import shutil
import requests
import configparser
from joblib import dump
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio import features, mask
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
json_headers = json.loads(config['DEFAULT']['json_headers'])
recreate_labs = True  # Always (re)create y-labels: something might have changed in the base layer


def get_point_data(raster_url, point):
    point_data = requests.get(url=raster_url + "/point/", headers=json_headers, params={"geom": point}).json()
    try:
        return point_data['results'][0]['value']
    except:
        return 0


def get_raster_data(raster_url, modifier, name, bbox, size):
    geotiff_name_msk = data_url + '/rasters/' + modifier + '/msk_' + name + '.tiff'

    if not Path(geotiff_name_msk).is_file():
        print("No masked .tiff-file found for layer {}; retrieving and extracting data".format(name))
        geotiff_name = data_url + '/rasters/' + modifier + '/' + name + '.tiff'
        if not Path(geotiff_name).is_file():
            bbox_str = ','.join(map(str, bbox))  # API requests a string-formatted list for some reason
            r = requests.get(url=raster_url + '/data/',
                             headers=json_headers,
                             params={"bbox": bbox_str, "format": 'geotiff', "width": size, "height": size})
            with open(geotiff_name, 'wb') as f:  # Write raster data to disk
                for data in r.iter_content(None):
                    f.write(data)
        # Don't ask me how the rest of this function works; it somehow does
        gdf = gpd.read_file(data_url + '/water_shp/invertedwatergang.shp')
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


def get_labels(model, modifier, name, copy):
    """
    Retrieve label data from .tiff

    :param model: str, 'expert_ref' or 'hist_buildings'
    :param name: name of the model, equal to model
    :param copy: layer data for feature burning
    :return: labels
    """
    geotiff_name = data_url + '/rasters/' + modifier + '/' + name + '.tiff'
    if model == 'expert_ref' and modifier.lower() in ['oc', 'ws']:
        if (not Path(geotiff_name).is_file()) or recreate_labs:
            print("(Re)creating layer {}.tiff.; retrieving expert scores".format(name))
            # if modifier.lower() == 'oc':
            #     dp_url = 'https://hhnk.lizard.net/api/v4/favourites/89c4db99-7a1a-4aa1-8abc-f89133d20d63/'
            # else: # modifier == ws
            #     dp_url = 'https://hhnk.lizard.net/api/v4/favourites/917100d2-7e3f-430f-a9d5-1fb42f5bb7d0/'
            # r = requests.get(url=dp_url, headers=json_headers)
            # data = r.json()['state']

            # json_url = data_url + '/datapoints_{}.json'.format(modifier)
            # if not Path(json_url).is_file():
            #     with open(json_url, 'w') as f:
            #         json.dump(data['geometries'], f)
            #
            # with open(json_url, 'r') as f:
            #     points = json.load(f)

            # scores = []
            # with open(data_url + '/expertscores_{}.csv'.format(modifier), 'r') as f:
            #     csvreader = csv.reader(f)
            #     next(csvreader)
            #     for row in csvreader:
            #         scores.append(float(row[3]))  # Mean
            #         # scores.append(float(row[4]))  # Std
            # gdf = gpd.GeoDataFrame.from_features(points, columns=['geometry', 'score'])
            # gdf['value'] = np.array(scores).flatten()
            #
            # print("gdf",gdf.columns, gdf)
            df = pd.read_csv(data_url + '/expertscores_{}.csv'.format(modifier), header=[0])
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
            # shapes = [(geom, int(value)) for geom, value in zip(gdf['geometry'], gdf['value'])]
            # shapes = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['value'])]
            shapes = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['Mean'])]

            copy_fn = data_url + '/rasters/' + modifier + '/' + copy + '.tiff'
            copy_f = rasterio.open(copy_fn)
            meta = copy_f.meta.copy()
            meta.update(compress='lzw')
            with rasterio.open(geotiff_name, 'w+', **meta) as out:
                out_arr = out.read(1)
                burned = features.rasterize(shapes=shapes, out=out_arr, transform=out.transform, fill=0)
                out.write_band(1, burned)
    elif model == 'hist_buildings' and modifier.lower() in ['purmer', 'schermerbeemster', 'purmerend', 'volendam']:
        if (not Path(geotiff_name).is_file()) or recreate_labs:
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
    else:  # Data is used for testing and does not have associated labels: copy a random layer
        copy_fn = data_url + '/rasters/' + modifier + '/' + copy + '.tiff'
        print(copy_fn, geotiff_name)
        shutil.copyfile(copy_fn, geotiff_name)

    with rasterio.open(geotiff_name) as f:  # Read raster data from disk
        data = f.read(1)

    return np.squeeze(data)


def get_fav_data(modifier, model, bbox, size):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/97ffd069-1da0-43f3-964e-cdded4a8565b/"
    r = requests.get(url=fav_url, headers=json_headers)
    data = r.json()['state']
    lon = np.linspace(min([bbox[0], bbox[2]]), max([bbox[0], bbox[2]]), size)
    lat = np.linspace(max([bbox[1], bbox[3]]), min([bbox[1], bbox[3]]), size)
    Y, X = np.meshgrid(lon, lat)  # This intentionally swapped: numpy
    col_names, combined_sources = ['lon', 'lat'], [X, Y]
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        col_names.append(layer['name'].strip())

        layerdata = np.array(get_raster_data(raster_url, modifier, layer['name'], bbox, size))
        combined_sources.append(layerdata)

    label_data = np.array(
        get_labels(model, modifier, name=model, copy=col_names[-2]))

    combined_sources.append(label_data)
    col_names.append(model.strip())
    return np.array(combined_sources), col_names


def create_df(modifier, bbox, size, model):
    if not os.path.exists(data_url + '/' + modifier):
        os.makedirs(data_url + '/' + modifier)
        os.makedirs(data_url + '/rasters/' + modifier)

    # Retrieve layer data based on a favourite instance
    extracted_data, col_names = get_fav_data(modifier, model, bbox, size)

    # NaN values are explicitly set to NaN instead of -9999
    extracted_data[extracted_data < -9000] = np.nan

    # (n_feat, w, h) -> (w x h, n_feat)
    data_extract2 = np.transpose(extracted_data, (1, 2, 0))
    data_extract3 = np.reshape(data_extract2, (-1, data_extract2.shape[-1]))

    # Normalize the features (without normalizing location and labels)
    ss = StandardScaler()
    extracted_loc = data_extract3[:, :2]
    extracted_feats = ss.fit_transform(data_extract3[:, 2:-1])
    extracted_labs = data_extract3[:, -1]
    extracted = np.concatenate((extracted_loc, extracted_feats, extracted_labs.reshape((-1, 1))), axis=1)

    df = pd.DataFrame(extracted, columns=col_names)

    # Change outlier values: more than 3 standard deviations from mean are set to 95% percentile
    for column in df.columns[2:-1]:
        threshold = np.nanpercentile(df[column], 95)
        mask = zscore(df[column], nan_policy='omit') > 3
        df.loc[mask, column] = threshold

    df.to_csv(data_url + '/' + modifier + "/" + model + ".csv", index=False)
    dump(ss, data_url + '/' + modifier + "/scaler.joblib")
    print("Data with shape {} saved to {}.csv".format(extracted.shape, modifier))
