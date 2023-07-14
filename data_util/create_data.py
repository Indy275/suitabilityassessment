import json
import os.path
import requests
import configparser
from joblib import dump
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio import features, mask
from shapely import Polygon
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('config.ini')

data_url = config['DEFAULT']['data_url']
fav_url = config['DEFAULT']['data_layers']
json_headers = json.loads(config['DEFAULT']['json_headers'])
recreate_labs = config.getboolean('DATA_SETTINGS', 'recreate_labs')
recreate_feats = config.getboolean('DATA_SETTINGS', 'recreate_feats')


def concoo(d0_loc):  # convert_coordinates
    # inProj = Proj(init='epsg:4326')  # world coordinates
    # outProj = Proj(init='epsg:28992')  # Dutch coordinates
    # d0_x0, d0_x1 = transform(inProj,outProj, d0_loc[0], d0_loc[1])
    # transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    d0_x0, d0_x1 = transformer.transform(d0_loc[1], d0_loc[0])
    return (d0_x0, d0_x1)


def create_bg(mod, bbox):
    bbox = [float(i[0:-1]) for i in bbox.split()]
    NH_w = 5.37725914616264 - 4.52690620343425  # MaxLng - minLng
    NH_h = 53.21427409446518 - 52.36877567446727  # maxLat - minLat
    poly_bbox = (concoo([bbox[0], bbox[3]]), concoo([bbox[2], bbox[3]]),concoo([bbox[2], bbox[1]]),concoo([bbox[0], bbox[1]]))
    poly = Polygon(poly_bbox)
    with rasterio.open(data_url+'/background.tif') as src:  # Read raster data from disk
        arr = src.read(1)
        orig_h = arr.shape[0]
        orig_w = arr.shape[1]
        raster_meta = src.meta
        clipped_dataset, out_transform = rasterio.mask.mask(src, [poly], crop=True)

    h = int(round(orig_h * abs(bbox[1]-bbox[3]) / NH_h))
    w = int(round(orig_w * abs(bbox[0]-bbox[2]) / NH_w))
    raster_meta.update({
        'height': h,
        'width': w,
        'crs': src.crs,
    })

    with rasterio.open(data_url+f'/{mod}/bg.tif', 'w', **raster_meta) as dst:  # Write data masked with watergangen
        dst.write(clipped_dataset)


def get_raster_data(raster_url, modifier, name, bbox, size):
    geotiff_name_msk = data_url + '/rasters/' + modifier + '/' + name + '_msk.tiff'

    if not Path(geotiff_name_msk).is_file() or recreate_feats:
        print("No masked .tiff-file found for layer {}; retrieving and extracting data".format(name))
        geotiff_name = data_url + '/rasters/' + modifier + '/' + name + '.tiff'
        if not Path(geotiff_name).is_file():
            bbox_str = ','.join(map(str, bbox))  # API requests a string-formatted list for some reason
            r = requests.get(url=raster_url + '/data/',
                             headers=json_headers,
                             params={"bbox": bbox_str, "format": 'geotiff', "width": size, "height": size})
            with open(geotiff_name, 'wb') as f:  # Write raster data from Lizard to disk
                for data in r.iter_content(None):
                    f.write(data)

        gdf = gpd.read_file(data_url + '/water_shp/invertedwatergangHHNK.shp')
        nodata_val = -9999
        with rasterio.open(geotiff_name) as src:  # Read raster data from disk

            raster_data = src.read()
            raster_data[raster_data < 0] = 0  # NaN is indicated with -9999, set to 0
            raster_meta = src.meta

        with rasterio.open(geotiff_name, 'w', **raster_meta) as dst:  # Write modified raster data to disk
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

        with rasterio.open(geotiff_name_msk, 'w', **raster_meta) as dst:  # Write data masked with watergangen
            dst.write(masked_data)

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

        if model == 'expert_ref':
            df = pd.read_csv(data_url + '/expertscores_{}.csv'.format(modifier), header=[0])
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lng, df.Lat))
            shapes = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['Value'])]

        elif model == 'hist_buildings':
            hist_bld_source = 'HHNKpre1900filtered'
            # if modifier.lower() in ['purmer', 'schermerbeemster', 'purmerend', 'volendam']:
            #     hist_bld_source = 'waterland1900min'
            # elif modifier.lower() in ['noordholland', 'noordhollandhires']:
            #     hist_bld_source = 'HHNKpre1900filtered'
            # else:
            #     print(f'There was an incompatibility issue: {model=} {modifier=}')
            gdf = gpd.read_file(data_url + f'/{hist_bld_source}.shp')
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
    """ Retrieve the data layers present in the favourite instance

    :param modifier: string indication of data source
    :param model: model labels
    :param bbox: bounding box
    :param size: size of data to be produced
    :param test: boolean indicating test data, to exclude labels
    :return:
    """
    r = requests.get(url=fav_url, headers=json_headers)
    data = r.json()['state']
    lng = np.linspace(min([bbox[0], bbox[2]]), max([bbox[0], bbox[2]]), size)
    lat = np.linspace(max([bbox[1], bbox[3]]), min([bbox[1], bbox[3]]), size)
    X, Y = np.meshgrid(lng, lat)
    col_names, combined_sources = ['Lng', 'Lat'], [X, Y]
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        col_names.append(layer['name'].strip())

        layerdata = np.array(get_raster_data(raster_url, modifier, layer['name'], bbox, size))
        combined_sources.append(layerdata)

    if not test:
        label_data = get_labels(model, modifier, copy=col_names[-2])
    else:  # test data -> no labels
        label_data = np.zeros((size, size))

    combined_sources.append(label_data)
    col_names.append(model.strip())

    data = np.array(combined_sources)

    # NaN values are explicitly set to NaN instead of -9999
    data[data < -9000] = np.nan

    # (n_feat, w, h) -> (w x h, n_feat)
    data = np.transpose(data, (1, 2, 0))
    data = np.reshape(data, (-1, data.shape[-1]))

    return data, col_names


def get_expert_data(modifier):
    expert_scores = pd.read_csv(data_url + f'/expertscores_{modifier.lower()}.csv', header=[0])
    exp_point_info = pd.read_csv(data_url + '/expert_point_info_{}.csv'.format(modifier), header=[0])
    point_info_scores = expert_scores.merge(exp_point_info, on='Point', how='left',
                                            suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    y = point_info_scores['Value'].to_numpy()
    point_info_scores.drop(['Value', 'Std', 'Point', 'Unnamed: 0'], axis=1, inplace=True, errors='ignore')
    col_names = list(point_info_scores.columns)
    col_names.append("expert_ref")
    X = point_info_scores.to_numpy()
    data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    return data, col_names


def create_df(modifier, bbox, size, model, test=False):
    if not os.path.exists(data_url + '/' + modifier):
        os.makedirs(data_url + '/' + modifier)
        os.makedirs(data_url + '/rasters/' + modifier)

    if model == 'expert_ref' and not test:
        data, col_names = get_expert_data(modifier)

    else:
        data, col_names = get_fav_data(modifier, model, bbox, size, test=test)

    # Change outlier values: cut-off at threshold
    df = pd.DataFrame(data, columns=col_names)
    thresholds = [7, 4, 80, .5, 2]  # 7m primary; 4m regional; 80mm subsidence; .5m waterdepth; 2m soil capacity
    for i, column in enumerate(df.columns[2:-1]):
        print(column, df[column].max())
        df.loc[df[column] > thresholds[i], column] = thresholds[i]

    # Normalize the features (without normalizing location and labels)
    data = df.to_numpy()
    ss = StandardScaler()
    feats_data = ss.fit_transform(data[:, :-1])
    labs_data = data[:, -1]
    data_conc = np.concatenate((feats_data, labs_data.reshape((-1, 1))), axis=1)

    df = pd.DataFrame(data_conc, columns=col_names)

    ref_std = 'testdata' if test else model
    df.to_csv(data_url + '/' + modifier + "/" + ref_std + ".csv", index=False)
    dump(ss, data_url + '/' + modifier + "/" + ref_std + "_scaler.joblib")
    print("Data with shape {} saved to {}/{}.csv".format(data_conc.shape, modifier, ref_std))

