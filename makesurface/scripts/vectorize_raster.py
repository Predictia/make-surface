import fiona, json, rasterio, click
from rasterio import features, Affine
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.geometry import polygon
from fiona.crs import from_epsg
import numpy as np
import tools
from scipy.ndimage import zoom
from scipy.ndimage.filters import median_filter, maximum_filter
import gdal
from osgeo import osr

def classifyManual(inArr, classArr):
    outRas = np.zeros(inArr.shape)
    breaks = {}
    for i in range(len(classArr)):
        breaks[i + 1] = float(classArr[i])
        if i<len(classArr)-1:
            outRas[np.where((inArr>=classArr[i]) & (inArr<=classArr[i+1]))] = i + 1
        else:
            outRas[np.where(inArr >= classArr[i])] = i + 1
    outRas[np.where(inArr.mask == True)] = np.nan
    breaks[0] = np.nan
    return outRas.astype(np.uint8), breaks

def zoomSmooth(inArr, smoothing, inAffine):
    zoomReg = zoom(inArr.data, smoothing, order=0, mode='nearest')
    zoomed = zoom(inArr.data, smoothing, order=1, mode='nearest')
    zoomMask = zoom(inArr.mask, smoothing, order=0, mode='nearest')
    zoomed[np.where(zoomed > inArr.max())] = inArr.max()
    zoomed[np.where(zoomed < inArr.min())] = inArr.min()
    inArr = np.ma.array(zoomed, mask=zoomMask)
    oaff = tools.resampleAffine(inAffine, smoothing)
    del zoomed, zoomMask
    return inArr, oaff

def vectorizeRaster(infile, outfile, classes, classfile, weight, nodata, smoothing, band, cartoCSS, axonometrize, nosimple, setNoData, nibbleMask, outvar):
    band = int(band)
    src = gdal.Open(infile)
    bandData = src.GetRasterBand(band)
    inarr = bandData.ReadAsArray()

    if (inarr is None) or (len(inarr) == 0):
        gdal.SetConfigOption('GDAL_NETCDF_BOTTOMUP','NO')
        src = gdal.Open(infile)
        bandData = src.GetRasterBand(band)
        inarr = bandData.ReadAsArray()
    
    oshape = np.shape(inarr)

    if len(src.GetProjectionRef())>0:
        new_cs = osr.SpatialReference()
        new_cs.ImportFromEPSG(4326)
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(src.GetProjectionRef())
        transform = osr.CoordinateTransformation(old_cs,new_cs)
    
    oaff = Affine.from_gdal(*src.GetGeoTransform())
    
    bbox = src.GetGeoTransform()
    nodata = None

    if type(bandData.GetNoDataValue()) == float:
        nodata = bandData.GetNoDataValue()

    if (type(setNoData) == int or type(setNoData) == float) and hasattr(inarr, 'mask'):
        inarr[np.where(inarr.mask == True)] = setNoData
        nodata = True

    nlat,nlon = np.shape(inarr)
    dataY = np.arange(nlat)*bbox[5]+bbox[3]
    dataX = np.arange(nlon)*bbox[1]+bbox[0]

    if len(src.GetProjectionRef())>0:
        ul = transform.TransformPoint(min(dataX),max(dataY))
        ll = transform.TransformPoint(min(dataX),min(dataY))
        ur = transform.TransformPoint(max(dataX),max(dataY))
        lr = transform.TransformPoint(max(dataX),min(dataY))
        simplestY1 = (abs(ul[1] - ll[1]) / float(oshape[0]))
        simplestY2 = (abs(ur[1] - lr[1]) / float(oshape[0]))
        simplestX1 = (abs(ur[0] - ul[0]) / float(oshape[1]))
        simplestX2 = (abs(lr[0] - ll[0]) / float(oshape[1]))
        simplest = 2*max(simplestX1,simplestY1,simplestX2,simplestY2)
    else:
        simplestY = ((max(dataY) - min(dataY)) / float(oshape[0]))
        simplestX = ((max(dataX) - min(dataX)) / float(oshape[1]))
        simplest = 2*max(simplestX,simplestY)

    if nodata == 'min':
        maskArr = np.zeros(inarr.shape, dtype=np.bool)
        maskArr[np.where(inarr == inarr.min())] = True
        inarr = np.ma.array(inarr, mask=maskArr)
        del maskArr
    elif type(nodata) == int or type(nodata) == float:
        maskArr = np.zeros(inarr.shape, dtype=np.bool)
        maskArr[np.where(inarr == nodata)] = True
        inarr[np.where(inarr == nodata)] = np.nan
        inarr = np.ma.array(inarr, mask=maskArr)
        del maskArr
    elif nodata == None or np.isnan(nodata) or nodata:
        maskArr = np.zeros(inarr.shape, dtype=np.bool)
        inarr = np.ma.array(inarr, mask=maskArr)
        del maskArr
    elif (type(nodata) == int or type(nodata) == float) and hasattr(inarr, 'mask'):
        nodata = True

    if nibbleMask:
        inarr.mask = maximum_filter(inarr.mask, size=3)

    if smoothing and smoothing > 1:
        inarr, oaff = zoomSmooth(inarr, smoothing, oaff)
    else:
        smoothing = 1

    with open(classfile, 'r') as ofile:
        classifiers = ofile.read().split(',')
        classRas, breaks = classifyManual(inarr, np.array(classifiers).astype(inarr.dtype))
    
    # filtering for speckling
    classRas = median_filter(classRas, size=2)

    # print out cartocss for classes
    if cartoCSS:
        for i in breaks:
            click.echo('[value = ' + str(breaks[i]) + '] { polygon-fill: @class' + str(i) + '}')

    if outfile:
        outputHandler = tools.dataOutput(True)
    else:
        outputHandler = tools.dataOutput()
    #polys = []
    #vals = []
    for i, br in enumerate(breaks):
        if i==0:
            continue
        tRas = (classRas == i).astype(np.uint8)
        if nodata:
            tRas[np.where(classRas == 0)] = 0

        for feature, shapes in features.shapes(np.asarray(tRas,order='C'),transform=oaff):
            if shapes == 1:
                featurelist = []
                for c, f in enumerate(feature['coordinates']):
                    if len(src.GetProjectionRef())>0:
                        for ix in range(len(f)):
                            px = transform.TransformPoint(f[ix][0],f[ix][1])
                            lst = list()
                            lst.append(px[0])
                            lst.append(px[1])
                            f[ix] = tuple(lst)
                    if len(f) > 3 or c == 0:
                        if axonometrize:
                            f = np.array(f)
                            f[:,1] += (axonometrize * br)
                        if nosimple:
                            poly = Polygon(f)
                        else:
                            poly = Polygon(f).simplify(simplest / float(smoothing), preserve_topology=True)
                            if c == 0:
                                poly = polygon.orient(poly,sign=-1.0)
                            else:
                                poly = polygon.orient(poly,sign=1.0)
                            featurelist.append(poly)
                if len(featurelist) != 0:
                    #polys.append(MultiPolygon(featurelist))
                    #vals.append(breaks[br])
                    outputHandler.out({
                        'type': 'Feature',
                        'geometry': mapping(MultiPolygon(featurelist)),
                        'properties': {
                            outvar: breaks[br]
                        }
                    })

    #for pa in range(0,len(polys)):
    #    for pb in range(0,len(polys)):
    #        if pa==pb:
    #            continue
    #        if polys[pa].contains(polys[pb]) & (polys[pa].area>polys[pb].area):
    #            try:
    #                polys[pa] = polys[pa].difference(polys[pb])
    #                print polys[pa].area
    #                print '---'
    #                break
    #            except:
    #                a = 1
    #
    #for pc in range(0,len(polys)):
    #    outputHandler.out({
    #        'type': 'Feature',
    #        'geometry': mapping(polys[pc]),
    #        'properties': {
    #            outvar: vals[pc]
    #        }
    #    })
    if outfile:
        with open(outfile, 'w') as ofile:
            ofile.write(json.dumps({
                "type": "FeatureCollection",
                "features": outputHandler.data
            }))
