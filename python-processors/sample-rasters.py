import fiona, argparse, json
from rasterio import features
import rasterio as rio
import numpy as np

parser = argparse.ArgumentParser(description='Create Density Surface from GeoJSON points.')

parser.add_argument('infile',
                    help='Input Raster')

parser.add_argument('inshp',
                    help='Input Shapefile')

parser.add_argument('outJSON',
                    help='Output JSON')

parser.add_argument('field',
                    help='Input Field to Aggregate')


args = parser.parse_args()

with rio.open(args.infile, 'r') as src:
    nir = src.read_band(4)
    oshape = src.shape
    otrans = src.transform

with fiona.open(args.inshp, 'r') as shp:
    fields = list(set(feat['properties'][args.field] for feat in shp))

with fiona.open(args.inshp, 'r') as shp:
    rasters = features.rasterize(
                    ((feat['geometry'], feat['properties'][args.field]) for feat in shp),
                    out_shape=oshape,
                    transform=otrans)

out = {}

for i in fields:
    valuearr = nir[np.where(rasters == i)]
    out[i] = valuearr.ravel().tolist();

with open(args.outJSON, 'w') as ofile:
    ofile.write(json.dumps(out, indent=4))