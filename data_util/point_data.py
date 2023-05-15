import numpy as np
import pandas as pd
import requests
import configparser
import itertools, string
import rasterio

config = configparser.ConfigParser()
config.read('config.ini')

data_url = 'C:\\Users\indy.dolmans\Documents\data'
json_headers = {"username": "__key__",
                "password": "de4OeXZf.KCvG8v1AJxLzsemHXqDBMdsxPPpCmPeA",
                "Content-Type": "application/json"}


def save_point_data(uuid):
    fav_url = "https://hhnk.lizard.net/api/v4/favourites/{}/".format(uuid)
    r = requests.get(url=fav_url, headers=json_headers).json()
    data = r['state']
    geoms = [geom['geometry']['coordinates'] for geom in data['geometries']]
    geomsx = [geom['geometry']['coordinates'][0] for geom in data['geometries']]
    geomsy = [geom['geometry']['coordinates'][1] for geom in data['geometries']]
    print(geoms)
    min_x = min(geomsx)
    min_y = min(geomsy)
    max_x = max(geomsx)
    max_y = max(geomsy)
    point_data, names = [], []
    for layer in data['layers']:
        raster_url = "https://hhnk.lizard.net/api/v4/rasters/{}".format(layer['uuid'])
        bbox = [min_x, min_y, max_x, max_y]
        bbox_str = ','.join(map(str, bbox))
        r = requests.get(url=raster_url + '/data/',
                         headers=json_headers,
                         params={"bbox": bbox_str, "format": 'geotiff', "width": 100, "height": 100})
        geotiff_temp = data_url + '/temp.tiff'
        with open(geotiff_temp, 'wb') as f:  # Write raster data to disk
            for d in r.iter_content(None):
                f.write(d)
        with rasterio.open(geotiff_temp) as f:  # Write raster data to disk
            raster_data = [x[0] for x in f.sample(geoms)]
        point_data.append(raster_data)
        names.append(layer['name'])
    print(point_data)
    pdata = pd.DataFrame(list(zip(geoms, point_data)), columns=['point'] + names)
    pdata.to_csv(data_url + "/datapoints_OC.csv", index=False)


uuid = 'f84055d0-95b0-4a0d-b4b0-162efe032094'
save_point_data(uuid)

