from pyproj import Transformer


def convert_coords(d0_loc):
    # inProj = Proj(init='epsg:4326')  # world coordinates
    # outProj = Proj(init='epsg:28992')  # Dutch coordinates
    # d0_x0, d0_x1 = transform(inProj,outProj, d0_loc[0], d0_loc[1])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992")
    # transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")
    d0_x0, d0_x1 = transformer.transform(d0_loc[1], d0_loc[0])
    return [d0_x0, d0_x1]