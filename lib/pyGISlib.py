"""
Read and write raster files and shapefiles, and perform simple GIS operations

"""
from __future__ import division
from __future__ import print_function

from builtins import input
from builtins import zip
from builtins import range
from past.utils import old_div
import os.path
import math
import pdb
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import shapely, shapely.geometry
try:
    import dbf_python
except ImportError:
    print('warning: failed to import dbf_python module, ' \
          'cannot save shapefile attributes and .dbf files')

try:
    import osgeo.osr
    import osgeo.gdal
    import osgeo.gdal_array
    import osgeo.gdalconst
except ImportError:
    print('warning, failed to import osgeo modules')

# import shapefile.py library
# see http://code.google.com/p/pyshp/
try:
    import shapefile
except ImportError:
    print('warning: failed to import shapefile module')
    print('get the python shapefile library at http://code.google.com/p/pyshp/')


__author__ = "Elco Luijendijk"
__copyright__ = "Copyright 2014-2017, Elco Luijendijk"
__license__ = "GPL"
__version__ = "0.3"
__email__ = "elco.luijendijk at gmail.com"
__status__ = "Development"

try:
    osgeo.gdal.UseExceptions()
except NameError:
    pass


def open_raster_file(filename):
    """
    open raster file, and return raster object
    
    Parameters
    ----------
    filename : string
        filename of raster

    Returns
    -------
    raster : osgeo.gdal raster object

    """

    if os.path.isfile(filename):
        raster = osgeo.gdal.Open(filename, osgeo.gdal.GA_ReadOnly)
    else:
        print('error, could not open raster file %s' % filename)
        return None

    return raster


def get_raster_info(raster):
    """
    get basic raster info
    
    Parameters
    -----------
    raster : osgeo.gdal raster object
    
    Returns
    -------
    dimensions : list
        raster size.
    origin : list
        raster origin.
    cellsize : list
        pixel size
    nodata: float
        nodata value
    projection : osgeo.osr.SpatialReference class
    
    """

    inband = raster.GetRasterBand(1)
    geotransform = raster.GetGeoTransform()
    nodata = inband.GetNoDataValue()
    origin = [geotransform[0], geotransform[3]]
    cellsize = [geotransform[1], geotransform[5]]
    dimensions = [inband.XSize, inband.YSize]

    projection = osgeo.osr.SpatialReference()
    projection.ImportFromWkt(raster.GetProjectionRef())

    return dimensions, origin, cellsize, nodata, projection


def read_part_of_raster(raster, offset, dimensions):
    """
    Read subset of raster file into numpy array
    
    Parameters
    ----------
    raster : osgeo.gdal raster object.
    offset : (int,int)
        x and y offset
    dimensions : (int,int)
        extent of raster window in x and y direction
    
    Returns
    -------
    raster_subset_array : array
         Raster data with shape (nx x ny)
         
    """

    raster_subset_array = raster.ReadAsArray(xoff=int(offset[0]),
                                             yoff=int(offset[1]),
                                             xsize=int(dimensions[0]),
                                             ysize=int(dimensions[1]))

    return raster_subset_array


def read_raster_file(filename, verbose=False, band_number=1):
    """
    Read gdal-compatible raster file and convert to numpy array
    
    Parameters
    ----------
    filename : string
        filename of gdal compatible raster file
    verbose : bool, optional
        verbose output
    
    Returns
    -------
    raster_array : array
        raster data
    dimensions : list
        x and y size of raster
    origin : list 
        coordinates of (0,0) point of raster
    cellsize : list
        cellsize of raster
    nodata : float
        nodata value
    projection : osgeo.osr.SpatialReference class

    """

    if os.path.isfile(filename):
        raster = osgeo.gdal.Open(filename, osgeo.gdal.GA_ReadOnly)
    else:
        print('error, could not open file %s' % filename)
        return None, None, None, None, None, None

    if verbose is True:
        print('\tnumber of raster bands:', raster.RasterCount)

    inband = raster.GetRasterBand(band_number)
    geotransform = raster.GetGeoTransform()
    dimensions = [inband.XSize, inband.YSize]
    nodata = inband.GetNoDataValue()
    origin = [geotransform[0], geotransform[3]]
    cellsize = [geotransform[1], geotransform[5]]
    projection = osgeo.osr.SpatialReference()
    projection.ImportFromWkt(raster.GetProjectionRef())

    if verbose is True:
        print('\torigin = (', geotransform[0], ',', geotransform[3], ')')
        print('\tpixel Size = (', geotransform[1], ',', geotransform[5], ')')
        print('\tdimensions: x= %s, y= %s' % (inband.XSize, inband.YSize))
        print('\tstart reading raster file')

    raster_array = raster.ReadAsArray()

    if verbose is True:
        print('\tfinished reading raster file')
        print('min,max data values = %0.1f - %0.1f' \
              % (raster_array.min(), raster_array.max()))

    return raster_array, dimensions, origin, cellsize, nodata, projection


def mask_raster(raster, nodata):
    """
    mask nodata values in a raster
    
    Parameters
    ----------
    raster : numpy array 
        raster filename
    nodata : float
        nodata value
    
    Returns
    -------
    masked_raster : masked numpy array
        masked numpy array
    
    """

    masked_raster = np.ma.masked_array(raster, raster == nodata)

    return masked_raster


def write_raster_file(filename, raster_array,
                      origin, cellsize, nodata,
                      nbands=1,
                      crs='EPSG:4326',
                      rasterformat='GTiff'):
    """
    Save a numpy array as a GDAL compatible raster file.
    
    Uses osgeo.gdal module to do all the real work.
    
    See this webpage for gdal raster formats:
        http://www.gdal.org/formats_list.html
    
    Parameters
    ----------
    filename : string 
        raster filename
    raster_array : array
        raster data
    origin : list
        x,y coordinate of (0,0) point of raster
    cellsize : list
        cellsizes
    nodata : float
        nodata value
    nbands : int, optional
        number of raster bands, default is 1
    crs= 'EPSG:4326' : string or osgeo.osr.SpatialReference object, optional
        spatial reference, default='EPSG:4326' ,i.e. lat/long WGS84 projection
    rasterformat='GTiff' : string, optional
        raster type, default is 'Gtiff'    
    """

    dataset = osgeo.gdal.GetDriverByName(rasterformat)
    dst_ds = dataset.Create(filename,
                            raster_array.shape[1],
                            raster_array.shape[0],
                            nbands,
                            osgeo.gdalconst.GDT_Float64)

    dst_ds.SetGeoTransform([origin[0], cellsize[0], 0.0,
                            origin[1], 0.0, cellsize[1]])

    for i in range(nbands):
        band_out = dst_ds.GetRasterBand(i + 1)
        if nodata:
            band_out.SetNoDataValue(float(nodata))

        # # save array to raster band
        # single band only:
        if len(raster_array.shape) == 2:
            osgeo.gdal_array.BandWriteArray(band_out, raster_array)
        # multiple bands:
        elif len(raster_array.shape) == 3:
            osgeo.gdal_array.BandWriteArray(band_out, raster_array[:, :, i])

        band_out.FlushCache()

    # user-defined projection
    if type(crs) == str:
        srs = osgeo.osr.SpatialReference()
        srs.SetFromUserInput(crs)
    # otherwise, use osgeo.osr.SpatialReference class:
    else:
        srs = crs

    dst_ds.SetProjection(srs.ExportToWkt())

    dst_ds = None

    return


def load_shapefile(shapefilename, read_records=False, read_bbox=False,
                   verbose=False, records_as_dataframe=False,
                   force_float=True):
    """
    Read point, line or polygon shapefiles
    
    Wrapper around shapefile.py library (http://code.google.com/p/pyshp/)
    
    Parameters
    ----------
    shapefilename : str 
        filename.
    read_records : bool, optional
        read shapefile records
    read_bbox : bool, optional
        read and return bounding box of shapes
    verbose : bool, optional 
        extra output
    records_as_dataframe : bool, optional
        if True, store the shapefile records in a pandas dataframe 
    force_float : bool, optional
        force a float datatype for dataframe columns wherever possible
   
    Returns
    -------
    shapexy : list
        nested lists containing (Nx2) numpy arrays of x and y coordinates of
        of shape segments
    fields : list, optional
        shape attribute field names, only returned if read_records=True and
        records_as_dataframe = False
    records : list, optional
        shape records, only returned if read_records= True and 
        records_as_dataframe = False
    df : pandas dataframe, optional
        pandas datafram containing all shapefile attributes. Only returned if
        read_records=True and records_as_dataframe=True
    bbox : list, optional
        shape bounding box. only returned if read_bbox = True
    
    """

    if not os.path.exists(shapefilename):
        raise IOError('File %s not found' % shapefilename)

    else:
        shapexy = []

        if verbose is True:
            print('processing %s' % shapefilename)

        shf = shapefile.Reader(shapefilename)
        shapes = shf.shapes()

        # read shapes one by one, reader fails for large shapefiles (?)
        if shapes is []:
            print('reading shapes one by one')
            go = True
            i = 0
            while go is True:
                print('shape %i' % i)
                try:
                    shapes.append(shf.shape())
                    print(shapes[-1])
                    i += 1
                except IOError:
                    go = False
            pdb.set_trace()

            # points shapefile:
        if shapes[0].shapeType == 1:
            shapexy_local = []
            for shape in shapes:
                shapexy_local.append(shape.points[0])
            shapexy.append(np.asarray(shapexy_local))

        # line or polygon shapefile
        elif shapes[0].shapeType == 3 or shapes[0].shapeType == 5:
            for shape in shapes:

                if shape.points is not []:  # <- skip empty shapes

                    shape_xy = np.asarray(shape.points)

                    if hasattr(shape, 'parts'):
                        parts = shape.parts
                        parts.append(shape_xy.shape[0])
                        shapexyp = []
                        for part_start, part_end in zip(parts[:-1], parts[1:]):
                            shapexyp.append(shape_xy[part_start:part_end])

                        shapexy.append(shapexyp)
                    else:
                        shapexy.append(shape_xy)

        # read bounding box
        if read_bbox is True:
            bbox = [shape.bbox for shape in shapes]

            # construct a pandas dataframe
        if records_as_dataframe is True:
            field_names = [sh[0] for sh in shf.fields[1:]]
            df = pd.DataFrame(np.array(shf.records()), columns=field_names)

            # force float datatype for dataframe whereever possible
            if force_float is True:
                for col in df.columns:
                    try:
                        df[col] = df[col].astype(float)
                    except (ValueError, TypeError, IndexError):
                        pass

        return_list = [shapexy]

        if read_records is True and records_as_dataframe is True:
            return_list.append(df)
        elif read_records is True and records_as_dataframe is False:
            return_list += [shf.fields, shf.records()]
        if read_bbox is True:
            return_list.append(bbox)

        return return_list


def find_point_location_in_raster(origin, cellsize, point_xy):

    """
    Find the location of a point in a raster

    Parameters
    ----------
    cellsize : list
        [x,y] list of pixel size of raster
    x : array
        x-coordinates point
    y : array
        y-coordinates point

    Returns
    -------
    xyo : array
        [x,y] array with x and y location of point origin in raster
    xysize : array
        [x,y] array of extent of point in raster units
    """

    # find min, max coordinates of point
    xy = np.array([old_div((point_xy[0] - origin[0]), cellsize[0]),
                   old_div((point_xy[1] - origin[1]), cellsize[1])])

    return xy


def get_polygon_location_in_raster(cellsize, polygon_x, polygon_y):

    """
    Find the origin and extent of a polygon in a raster
    
    Parameters
    ----------
    cellsize : list
        [x,y] list of pixel size of raster
    polygon_x : array
        x-coordinates polygon
    polygon_y : array
        y-coordinates polygon
    
    Returns
    -------
    xyo : array
        [x,y] array with x and y location of polygon origin in raster
    xysize : array
        [x,y] array of extent of polygon in raster units
    """

    # check whether raster is upside down:
    np_origin = [np.min, np.min]
    np_size = [np.max, np.max]
    if np.sign(cellsize[0]) == -1:
        np_origin[0] = np.max
        np_size[0] = np.min
    if np.sign(cellsize[1]) == -1:
        np_origin[1] = np.max
        np_size[1] = np.min

    # find min, max coordinates of polygon
    xyo = np.array([np_origin[0](polygon_x), np_origin[1](polygon_y)])
    xysize = np.array([np_size[0](polygon_x) - xyo[0],
                       np_size[1](polygon_y) - xyo[1]])

    return xyo, xysize


def get_raster_coordinates(dimensions, cellsize, origin):

    """
    
    Parameters
    ----------
    dimensions : list
        [nx, ny] size of input raster
    cellsize : list
        [x,y] cell size of raster
    origin : list
        [x,y] coordinates of origin of raster
    
    Returns
    -------
    raster_x : array
        2D array of x-coordinates raster
    raster_y : array
        2D array of y-coordinates raster
    
    """

    # create raster with elevation grid coordinates

    xcoords = (np.arange(dimensions[0])) * cellsize[0] + origin[0] + cellsize[0] * 0.5
    ycoords = (np.arange(dimensions[1])) * cellsize[1] + origin[1] + cellsize[1] * 0.5
    raster_x, raster_y = np.meshgrid(xcoords, ycoords)

    return raster_x, raster_y


def clip_masked_raster_to_polygon(input_raster, raster_x, raster_y,
                                  polygon_x, polygon_y):
    """
    Clip a raster to a polygon
    
    Parameters
    ----------
    input_raster : masked array
    raster_x : array
        x coordinates of raster
    raster_y : array 
        y coordinates of raster
    polygon_x : array
        x coordinates of polygon
    polygon_y : array 
        y coordinates of polygon
        
    
    Returns
    -------
    output_raster : masked array
        raster with updated mask
        
    """

    polygon = shapely.geometry.Polygon(list(zip(polygon_x, polygon_y)))

    raster_pts = shapely.geometry.MultiPoint(list(zip(raster_x.ravel(),
                                                 raster_y.ravel())))

    # find raster points that are located in polygon
    pt_in_poly = [polygon.intersects(pt) for pt in raster_pts]
    raster_in_poly = np.resize(np.array(pt_in_poly), raster_x.shape)

    # clip elevation data to polygon
    input_raster.mask[raster_in_poly == False] = True

    return input_raster


def simplify_shapefiles(shapex, shapey):

    """
    simplify shapefiles, merge muliple segments to single segment shapes
    
    Parameters
    ----------
    shapex : list
        nested list of x coordinates of shapes 
        [[shape1_segment1, shape1_segment2], [shape2_segmen1], etc...]
        where shape1_segment1 is a numpy array of the x-coordinates
    shapey : list 
        nested list of y coordinates of shapes 
    
    Returns
    -------
    shapexm : list
        list of numpy arrays of x-coordinates of shapes, where all segments 
        are merged
        [shape1, shape2, etc...]
        shape1 is a nx1 numpy array of the x-coordinates of the shape
    shapeym : list 
        list of numpy arrays of y-coordinates of shapes

    """

    shapexm = [[]] * len(shapex)
    shapeym = [[]] * len(shapey)

    for i, shapexi, shapeyi in zip(itertools.count(), shapex, shapey):
        if len(shapexi) == 1 and len(shapeyi) == 1:
            shapexm[i] = shapexi[0]
            shapeym[i] = shapeyi[0]
        elif len(shapexi) > 1 and len(shapeyi) > 1:
            shapexa = np.zeros(0)
            shapeya = np.zeros(0)

            for shapexii, shapeyii in zip(shapexi, shapeyi):
                shapexa = np.append(shapexa, shapexii)
                shapeya = np.append(shapeya, shapeyii)

            shapexm[i] = shapexa
            shapeym[i] = shapeya

    return shapexm, shapeym


def update_dbf_file(dbf_file, dbf_output_file, dataframe,
                    nodata_value=-99999, record_lenght=40,
                    record_decimals=5,
                    check_if_file_exists=True,
                    final_columns=None):

    """
    read a dbf file, add columns and records from a pandas dataframe and 
    write to a new dbf file
    
    warning: for now only floating point data is supported
    
    support for other types of data such as integers, strings etc. is planned
    
    Parameters
    ----------
    dbf_file : string
        name of existing dbf file
    dbf_output_file
        name of new dbf file with updated records
    dataframe : pandas.DataFrame instance
        name of pandas dataframe which contains the fields & records that will
        be appended to the existing dbf file
    nodata_value: float, optional
        value to use for non-float or numpy.nan data. default is -99999
    record_lenght : int, optional
        length of new field, default is 40
    record_decimals : int, optional
        number of decimals for new fields, default is 5
    check_if_file_exists : bool, optional
        check if the output dbf file already exists and prompt user to overwrite 
        file
    final_columns : list
        list containing the column names to include in the dbf file. Default is None. If None then all the columns in
        the existing dbf file and the new columns in the dataframe are added

    """

    # read existing dbf file
    print('reading existing dbf file %s' % dbf_file)
    fin = open(dbf_file, 'rb')
    db = list(dbf_python.dbfreader(fin))
    fin.close()

    fieldnames, fieldspecs, records = db[0], db[1], db[2:]

    # and add new fields
    n_new_cols = len(dataframe.columns)
    field_types = ['N'] * n_new_cols
    for col, field_type in zip(dataframe.columns, field_types):
        fieldnames.append(col[:10])
        fieldspecs.append((field_type, record_lenght, record_decimals))

    # populating records
    print('adding records to new .dbf file')
    modified_records = []
    for db_row, record, df_row in zip(itertools.count(),
                                                 records,
                                                 dataframe.iterrows()):

        # add new records from pandas dataframe to dbf file:
        for colno, col in enumerate(dataframe.columns):

            newdat = df_row[1][colno]

            if np.isnan(float(newdat)) == False:
                record.append(float(newdat))
            else:
                record.append(float(nodata_value))
                #print 'warning, found nodata'
                # record.append('')

        modified_records.append(record)

    #
    if final_columns is not None:
        field_ids = [fieldnames.index(fc) for fc in final_columns]

        recs_new = []
        for r in modified_records:
            rn = [r[i] for i in field_ids]
            recs_new.append(rn)

        fieldspecs_new = [fieldspecs[i] for i in field_ids]

        fieldnames = final_columns
        fieldspecs = fieldspecs_new
        modified_records = recs_new
        
    # saving new dbf file
    if check_if_file_exists is True and os.path.exists(dbf_output_file):

        print('file %s exists already, overwrite? (y/n)' % dbf_output_file)

        if 'y' in input():
            print('saving %s' % dbf_output_file)
            fout = open(dbf_output_file, 'w')
            dbf_python.dbfwriter(fout, fieldnames, fieldspecs, modified_records)
            fout.close()
    else:
        print('saving %s' % dbf_output_file)
        fout = open(dbf_output_file, 'w')
        dbf_python.dbfwriter(fout, fieldnames, fieldspecs, modified_records)
        fout.close()

    return


def convert_prj_to_proj4(prj_file):

    """
    read esri .prj file and return the projection in proj4 format 
    
    found on 
    http://stackoverflow.com/questions/3149112/how-can-i-proj4-details-from-the-shape-files-prj-file
    
    Parameters
    ----------
    prj_file : string
        filename of .prj file
    
    Returns
    -------
    proj_export : string
        projection in proj4 format
    
    """

    fin = open(prj_file, 'r')
    prj_text = fin.read()
    fin.close()

    srs = osgeo.osr.SpatialReference()
    if srs.ImportFromWkt(prj_text):
        raise ValueError("Error importing PRJ information from: %s" % prj_file)
    proj_export = srs.ExportToProj4()

    return proj_export


def get_polygon_origin(cellsize, polygon_xy):

    """
    
    Parameters
    ----------
    
    cellsize : array
        size of 1 grid cell in raster
    polygon_xy : array
        (Nx2) array with polygon coordinates
    
    Returns
    -------
    xyo : array
        polygon origin in raster
    xysize : array
        polygon size
    
    """

    # check whether raster is upside down:
    np_origin = [np.min, np.min]
    np_size = [np.max, np.max]
    if np.sign(cellsize[0]) == -1:
        np_origin[0] = np.max
        np_size[0] = np.min
    if np.sign(cellsize[1]) == -1:
        np_origin[1] = np.max
        np_size[1] = np.min

    # find min, max coordinates of watershed
    xyo = np.array([np_origin[0](polygon_xy[:, 0]),
                    np_origin[1](polygon_xy[:, 1])])
    xysize = np.array([np_size[0](polygon_xy[:, 0]) - xyo[0],
                       np_size[1](polygon_xy[:, 1]) - xyo[1]])

    return xyo, xysize


def get_raster_data_in_polygon(raster, dimensions, origin, cellsize, nodata,
                               polygon_xy, bbox=None, buffer=None):
    """
    Find part of raster that is located within a polygon and return a masked
    array of the raster values within the bounding box of the polygon
    
    Parameters
    ----------
    raster : gdal raster object
        raster object
    dimensions : array
        size of raster
    origin : array      
        [x,y] array with x and y cooridnate of the origin fo the raster
    cellsize : array
        [x,y] array of size of 1 raster cell
    nodata : float
        nodata value
    polygon_xy : array 
        (Nx2) numpy arrays of polygon coordinates
    
    Returns
    -------
    zm : array
        2-dimensional masked array with the raster values
    xcm : array
        2-dimensional masked array with the x-coordinates of the centre of each
        raster cell
    ycm : array
        2-dimensional masked array with the y coordinates of the centre of each
        raster cell
    
    """

    xyo, xysize = get_polygon_origin(cellsize, polygon_xy)

    #
    if buffer is not None:

        if xysize[1] > 0:
            xyo[0] -= buffer
            xyo[1] -= buffer
            xysize[0] += 2 * buffer
            xysize[1] += 2 * buffer

        else:
            xyo[0] -= buffer
            xyo[1] += buffer
            xysize[0] += 2 * buffer
            xysize[1] -= 2 * buffer

        #print 'applying buffer of %0.2e for polygon bnds' % buffer

    # calculate location of watershed in raster
    xyr_offset = np.floor((old_div((xyo - origin), cellsize))).astype(int)
    xyr_extent = np.ceil((old_div(((xyo + xysize) - origin), cellsize))).astype(int)
    xyr_size = xyr_extent - xyr_offset

    raster_origin = xyr_offset * cellsize + origin

    # check if watershed located within elevation raster
    if xyr_offset.min() < 0 or (xyr_offset[0] + xyr_size[0]) > dimensions[0] or \
                    (xyr_offset[1] + xyr_size[1]) > dimensions[1]:
        print('warning, polygon located outside raster, no elevation data read')
        return None, None, None

    # load elevation data within polygon bounding box:
    z = read_part_of_raster(raster,
                            xyr_offset,
                            xyr_size)

    # mask nodata values
    zm = np.ma.MaskedArray(z, z == nodata, fill_value=nodata)

    # create raster with coordinates of midpoint of raster cells
    xc = (np.arange(xyr_size[0]) + 0.5) * cellsize[0] + raster_origin[0]
    yc = (np.arange(xyr_size[1]) + 0.5) * cellsize[1] + raster_origin[1]
    xcm, ycm = np.meshgrid(xc, yc)

    return zm, xcm, ycm


def draw_shapely_object(input_shape, ax=None, **kwargs):

    """
    draw a shapely object

    Parameters
    ----------
    shape : shapely geometry or list of shapely geometries
        shapely geometry, can be any type (multi-) point, line or polygon
    ax : matplotlib axis instance
        defaults uses current axis
    kwargs :
        any matplotlib / pylab arguments to pass to scatter, fill or plot
        function, like color, linewidth, etc...

    Returns
    -------
    leg : matplotlib PathCollection

    """

    if ax is None:
        ax = pl.gca()

    if type(input_shape) == list:
        shapes = input_shape

    else:
        shapes = [input_shape]

    for shape in shapes:

        if shape.type == 'MultiPoint':
            pts = np.array([shape_i.coords[0] for shape_i in shape])
            leg = ax.scatter(pts, **kwargs)

        elif shape.type == 'MultiLineString':
            for line in shape:
                xy = np.array(line.xy)
                leg = ax.plot(xy[0], xy[1], **kwargs)

        elif shape.type == 'MultiPolygon':
            for polygon in shape:
                xy = np.array(polygon.exterior.xy)
                leg = ax.fill(xy[0], xy[1], **kwargs)

        elif shape.type == 'Point':
            leg = ax.scatter(shape.xy[0], shape.xyp[1], **kwargs)

        elif shape.type == 'LineString':
            xy = np.array(shape.xy)
            leg = ax.plot(xy[0], xy[1], **kwargs)

        elif shape.type == 'Polygon':
            xy = np.array(shape.exterior.xy)
            leg = ax.plot(xy[0], xy[1], **kwargs)

    return leg


def find_utm_zone(lon, lat):

    """

    find the utm zone for lat/long coordinates

    :param lon:
    :param lat:
    :return:
    """

    zone = int(math.floor(old_div((lon + 180.0), 6)) + 1)

    bands = "CDEFGHJKLMNPQRSTUVWXX"

    band = bands[int(math.floor(old_div((lat+80),8)))]

    return zone, band
