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
    geotiff_name_msk = data_url + '/rasters/' + modifier + '/' + name + '_msk.tiff'

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


def get_labels(model, modifier, copy):
    """
    Get target labels, either from expert judgement or historic buildings.
    Open .tiff-file with labels if existing, otherwise create the file first.

    :param model: str, 'expert_ref' or 'hist_buildings'
    :param modifier:
    :param copy: layer data for feature burning
    :return: labels
    """
    geotiff_name = data_url + '/rasters/' + modifier + '/' + model + '.tiff'
    if (not Path(geotiff_name).is_file()) or recreate_labs:  # if it doesn't yet exist, or if we want to recreate anyway
        print(f'(Re)creating layer; retrieving {model} labels')

        if model == 'expert_ref' and modifier.lower() in ['oc', 'ws']:
            df = pd.read_csv(data_url + '/expertscores_{}.csv'.format(modifier), header=[0])
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lng, df.Lat))
            shapes = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['Value'])]

        elif model == 'hist_buildings' and modifier.lower() in ['purmer', 'schermerbeemster', 'purmerend', 'volendam']:
            gdf = gpd.read_file(data_url + "/waterland1900min.shp")
            gdf.to_crs(crs='EPSG:4326', inplace=True)
            gdf['value'] = 1
            shapes = ((geom, value) for geom, value in zip(gdf['geometry'], gdf['value']))
        else:
            print(f'There was an incompatibility issue: {model=} {modifier=}')

        copy_fn = data_url + '/rasters/' + modifier + '/' + copy + '.tiff'
        copy_f = rasterio.open(copy_fn)
        meta = copy_f.meta.copy()
        meta.update(compress='lzw')

        with rasterio.open(geotiff_name, 'w+', **meta) as out:
            out_arr = out.read(1)
            burned = features.rasterize(shapes=shapes, out=out_arr, transform=out.transform, fill=0)
            out.write_band(1, burned)

    with rasterio.open(geotiff_name) as f:  # Read raster data from disk
        data = f.read(1)

    return np.array(np.squeeze(data))


def get_fav_data(modifier, model, bbox, size, test):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/97ffd069-1da0-43f3-964e-cdded4a8565b/"
    r = requests.get(url=fav_url, headers=json_headers)
    data = r.json()['state']
    lng = np.linspace(min([bbox[0], bbox[2]]), max([bbox[0], bbox[2]]), size)
    lat = np.linspace(max([bbox[1], bbox[3]]), min([bbox[1], bbox[3]]), size)
    Y, X = np.meshgrid(lng, lat)  # This intentionally swapped: numpy
    col_names, combined_sources = ['Lng', 'Lat'], [X, Y]
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        col_names.append(layer['name'].strip())

        layerdata = np.array(get_raster_data(raster_url, modifier, layer['name'], bbox, size))
        combined_sources.append(layerdata)

    if not test:
        label_data = get_labels(model, modifier, copy=col_names[-2])
    else:  # test data -> no labels
        label_data = np.zeros((size,size))

    combined_sources.append(label_data)
    col_names.append(model.strip())
    return np.array(combined_sources), col_names


def create_df(modifier, bbox, size, model, test=False):
    if not os.path.exists(data_url + '/' + modifier):
        os.makedirs(data_url + '/' + modifier)
        os.makedirs(data_url + '/rasters/' + modifier)

    if modifier.lower().startswith(('ws', 'oc')) and model == 'expert_ref' and not test:
        if modifier[-3:] == 'all':
            all_exp_scores = pd.read_csv(data_url + f'/expertscoresall_{modifier[:-3]}.csv', header=[0])
        else:
            all_exp_scores = pd.read_csv(data_url + f'/expertscores_{modifier}.csv', header=[0])
        exp_point_info = pd.read_csv(data_url + '/expert_point_info_{}.csv'.format(modifier), header=[0])
        point_info_scores = all_exp_scores.merge(exp_point_info, on='Point', how='left',
                                                 suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        y = point_info_scores['Value'].to_numpy()
        point_info_scores.drop(['Value', 'Std', 'Point', 'Unnamed: 0'], axis=1, inplace=True, errors='ignore')
        col_names = list(point_info_scores.columns)
        col_names.append("expert_ref")
        X = point_info_scores.to_numpy()
        data = np.concatenate((X, y.reshape((-1, 1))), axis=1)

    else:
        # Retrieve layer data based on a favourite instance
        data, col_names = get_fav_data(modifier, model, bbox, size, test=test)

        # NaN values are explicitly set to NaN instead of -9999
        data[data < -9000] = np.nan

        # (n_feat, w, h) -> (w x h, n_feat)
        data = np.transpose(data, (1, 2, 0))
        data = np.reshape(data, (-1, data.shape[-1]))

    # Normalize the features (without normalizing location and labels)
    ss = StandardScaler()
    loc_data = data[:, :2]
    feats_data = ss.fit_transform(data[:, 2:-1])
    labs_data = data[:, -1]
    data_conc = np.concatenate((loc_data, feats_data, labs_data.reshape((-1, 1))), axis=1)

    df = pd.DataFrame(data_conc, columns=col_names)

    # Change outlier values: more than 3 standard deviations from mean are set to 95% percentile
    for column in df.columns[2:-1]:
        threshold = np.nanpercentile(df[column], 95)
        mask = zscore(df[column], nan_policy='omit') > 3
        df.loc[mask, column] = threshold

    if not test:
        df.to_csv(data_url + '/' + modifier + "/" + model + ".csv", index=False)
    else:
        df.to_csv(data_url + '/' + modifier + "/testdata.csv", index=False)

    dump(ss, data_url + '/' + modifier + "/scaler.joblib")
    print("Data with shape {} saved to {}.csv".format(data_conc.shape, modifier))


