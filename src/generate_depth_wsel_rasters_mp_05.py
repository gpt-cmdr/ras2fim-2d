# Script 05 - From the HEC-RAS output, create floodplain rasters of the water
# surface elevation (WSEL) and depth for the simulation.
#
# This requires as HEC-RAS 2D model from which the following inputs are provided:
#  (1) plan file where the run has been executed
#  (2) terrain that will be used to compute the flood inundation raster stack
#  (3) hydraulic results geopackage (from previous steps)
#  (4) directory to write the output rasters
#
# Created by: Andy Carter, PE
# Created - 2024.07.08

# ************************************************************
import h5py
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, mapping, shape
import geopandas as gpd
import pandas as pd
import numpy as np
import os

import rasterio
from rasterio.transform import from_origin
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import shapes
import rasterio.mask

from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import configparser
import multiprocessing as mp


#import concurrent.futures
import argparse
import time
import datetime
import warnings
import json
import shutil
import tempfile
from typing import NamedTuple

import mapping_aoi
import rust_mapping
# ************************************************************


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist" % arg)
    else:
        # File exists so return the directory
        return arg
        return open(arg, 'r')  # return an open file handle
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ------------------------
def fn_get_group_names(hdf5_file_path, group_path):
    """
    Retrieve the names of groups within a specified HDF5 file under a given group path.

    Parameters:
    hdf5_file_path (str): The file path to the HDF5 file.
    group_path (str): The path to the group whose subgroups' names are to be retrieved.

    Returns:
    list or None: A list containing the names of groups found under the specified group path. 
                  Returns None if the group path does not exist in the HDF5 file.
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as hdf_file:
            # Check if the specified group path exists
            if group_path in hdf_file:
                group = hdf_file[group_path]

                # Extract names of HDF5 Group objects
                group_names = [name for name in group if isinstance(group[name], h5py.Group)]

                return group_names
            else:
                print(f"Group '{group_path}' not found in the HDF5 file.")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
# ------------------------


# ~~~~~~~~~~~~~~~~~~~~
def fn_gdf_cell_centerpoints(hdf_file_path,list_unique_indices_sorted, gdf_cells_wsel):

    # Specify the HDF5 file path and group path
    group_path = '/Geometry/2D Flow Areas/'

    # Get names of HDF5 Group objects in the specified group
    list_group_names = fn_get_group_names(hdf_file_path, group_path)

    # for now... lets assume that there is only one 2D area
    str_cell_info_path = '/Geometry/2D Flow Areas/Cell Info'

    str_cell_center_point_path = group_path + list_group_names[0] + '/' + 'Cells Center Coordinate'

    # From the plan HDF, get the cells center coordinates (not the centroid)
    with h5py.File(hdf_file_path, 'r') as hdf_file:
        arr_cell_center_coords = hdf_file[str_cell_center_point_path][:]

    # Filter the array to those cells that are wet
    arr_center_coords_wet_cells = arr_cell_center_coords[list_unique_indices_sorted]

    geometry = [Point(x, y) for x, y in arr_center_coords_wet_cells]

    # Create a GeoDataFrame
    gdf_center_points = gpd.GeoDataFrame(geometry=geometry, columns=['geometry'], crs=gdf_cells_wsel.crs)

    # create a new coloumn that contains the cell index
    gdf_center_points['cell_idx'] = list_unique_indices_sorted
    
    return (gdf_center_points)
# ~~~~~~~~~~~~~~~~~~~~


# -------------------------
def fn_get_raster_resolution(str_dem_path):
    from osgeo import gdal, gdalconst
    
    # Get the resolution of the ground DEM
    # Open the raster file
    dataset = gdal.Open(str_dem_path)

    # Get raster resolution
    flt_pixel_width = dataset.GetGeoTransform()[1]

    # Close the dataset raster file
    dataset = None  # Release the reference to the dataset
    
    return(flt_pixel_width)
# -------------------------


# .....................
def fn_wet_cells_per_run(str_wsel_col_name, gdf_cells_wsel, gdf_center_points):

    # from geodataframe, create dataframe from only the 'cell_idx' and 'wsel' coloumns
    df_wsel = gdf_cells_wsel[['cell_idx', str_wsel_col_name]]

    # left join the WSEL per cell to populate the geodataframe of cells with WSEL
    gdf_wsel_wet_cells_pnt = pd.merge(gdf_center_points, df_wsel, on='cell_idx', how='left')

    # Drop rows where "wsel_max_1" is NaN
    gdf_wsel_wet_cells_cleaned_pnt = gdf_wsel_wet_cells_pnt.dropna(subset=[str_wsel_col_name]).copy()

    # Drop rows where "wsel_max_1" is NaN
    gdf_wsel_wet_cells_cleaned_ar = gdf_cells_wsel.dropna(subset=[str_wsel_col_name]).copy()

    # set constant arbitrary value used to dissolve geometry
    gdf_wsel_wet_cells_cleaned_ar.loc[:, 'val'] = 1

    # create a geodataframe of the merged cells
    gdf_dissolved = gdf_wsel_wet_cells_cleaned_ar.dissolve(by='val')

    # TODO - 2024.06.12 - Is the first polygon correct?
    # get shapely geometry of the first merged polygon
    shp_wet_cells = gdf_dissolved.geometry.iloc[0]

    # Convert Shapely polygon to GeoJSON-like format
    geom = mapping(shp_wet_cells)
    
    return (geom, gdf_dissolved, gdf_wsel_wet_cells_cleaned_pnt, gdf_wsel_wet_cells_cleaned_ar)
# .....................


# -------------
def fn_compute_convex_hull(points):
    hull = ConvexHull(points)
    return [points[vertex] for vertex in hull.vertices]

def fn_calculate_vertex_points(polygon):
    list_points = []
    coords = polygon.exterior.coords
    num_coords = len(coords)
    for i in range(num_coords):  # Iterate over vertex
        x, y = coords[i]
        list_points.append(Point(x, y))
    return list_points

def fn_is_point_inside_convex_hull(point, hull_points):
    hull_polygon = Polygon(hull_points)
    return hull_polygon.contains(point)

# Define a function to merge points within a certain distance threshold
def fn_merge_points_within_threshold(gdf, threshold, str_wsel_col_name):
    merged_points = []
    grouped = gdf.groupby(gdf.geometry.apply(lambda x: (round(x.x, 2), round(x.y, 2))))
    
    for _, group in grouped:
        if len(group) > 1:
            merged_point = Point(group.geometry.x.mean(), group.geometry.y.mean())
            avg_wsel = group[str_wsel_col_name].mean()  # Calculate the average of 'wsel' column
            merged_points.append({'geometry': merged_point, str_wsel_col_name: avg_wsel})
        else:
            merged_points.append({'geometry': group.iloc[0].geometry, str_wsel_col_name: group.iloc[0][str_wsel_col_name]})
    
    return gpd.GeoDataFrame(merged_points)
# -------------

# --------------
def fn_nudge_raster(str_raster_path, str_output_path, str_crs, flt_res):
    flt_res = float(flt_res)  # Ensure the resolution is a float

    # Open the original raster
    with rasterio.open(str_raster_path) as src:
        # Calculate the transformation for the desired CRS
        
        # TODO -- there is a Linux PROJ issue with "str_crs" -- 2025.01.30
        
        transform, width, height = calculate_default_transform(
            src.crs, str_crs, src.width, src.height, *src.bounds
        )
        
        # Debugging - 2025.01.30
        #print(f"Source CRS: {src.crs}")
        #print(f"Target CRS: {str_crs}")
        
        # Update transform to match the desired resolution
        transform = from_origin(
            round(transform.c / flt_res) * flt_res,  # Nudge X to nearest multiple of flt_res
            round(transform.f / flt_res) * flt_res,  # Nudge Y to nearest multiple of flt_res
            flt_res, flt_res
        )

        # Prepare metadata for the output raster
        profile = src.profile
        profile.update({
            'crs': str_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',  # Ensure output raster is float32 for NaNs
            'compress': 'lzw',    # Compression if needed
            'nodata': np.nan      # Set nodata value to NaN
        })

        # Create the output raster
        with rasterio.open(str_output_path, 'w', **profile) as dst:
            # Loop through each band
            for i in range(1, src.count + 1):
                # Create an empty array for the output band with NaN
                output_band = np.full((height, width), np.nan, dtype='float32')

                # Reproject source data to the output band
                reproject(
                    source=rasterio.band(src, i),
                    destination=output_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=str_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata  # Treat src_nodata as NaN
                )

                # Update the output raster band only with valid data
                dst.write(output_band, i)

# --------------


# ~~~~~~~~~~~~~
def fn_set_nan(str_original_raster_path):

    str_intermediate_path = str_original_raster_path[:-4] + '_temp.tif'

    # Open the original raster file
    with rasterio.open(str_original_raster_path) as src:
        profile = src.profile
        profile['nodata'] = np.nan  # Set NoData value to NaN
        data = src.read(1)  # Read the first band

        # Convert NaN values to the specified NoData value
        data_with_nodata = np.where(np.isnan(data), np.nan, data)

        # Create a new raster file with NoData set to NaN
        with rasterio.open(str_intermediate_path, 'w', **profile) as dst:
            dst.write(data_with_nodata, 1)

        #print("New raster with NoData set to NaN has been created.")

        return(str_intermediate_path)
# ~~~~~~~~~~~~~


# --------------------------------
class PreparedInterpolationInputs(NamedTuple):
    """Global interpolation inputs shared by the python and rust engines."""
    x: np.ndarray        # sample x: wet centres (sorted cell_idx), then merged extras
    y: np.ndarray
    z: np.ndarray        # sample WSE in the same order
    x_axis: np.ndarray   # exact np.arange query axes (effective step != pixel size)
    y_axis: np.ndarray
    xmin: float          # raster 01 affine origin and pixel size
    ymin: float
    x_res: float
    y_res: float


def fn_prepare_interpolation_inputs(gdf_wsel_wet_cells_cleaned_pnt,
                                    gdf_wsel_wet_cells_cleaned_ar,
                                    str_wsel_col_name,
                                    gdf_dissolved,
                                    flt_pixel_width):
    """
    Upstream Step 05 sample preparation: wet cell centres plus the cell
    vertices outside their convex hull (duplicates merged by rounded XY with
    mean WSE), and the np.arange query axes over the wet-union bounds.
    Always global - an AOI never crops these inputs.
    """

    created_points = []
    list_cell_wsel = []

    # Wrap Convex Hull
    convex_hull_points = fn_compute_convex_hull(gdf_wsel_wet_cells_cleaned_pnt.geometry.apply(lambda point: (point.x, point.y)).tolist())

    # Iterate through cell polygons and perform checks
    for _, cell_polygon in gdf_wsel_wet_cells_cleaned_ar.iterrows():
        vertex_points = fn_calculate_vertex_points(cell_polygon.geometry)
        for v_point in vertex_points:
            if not fn_is_point_inside_convex_hull(v_point, convex_hull_points):
                created_points.append(v_point)
                list_cell_wsel.append(cell_polygon[str_wsel_col_name])

    # Create a new GeoDataFrame from created points
    gdf_additional_points = gpd.GeoDataFrame(geometry=created_points)

    gdf_additional_points[str_wsel_col_name] = list_cell_wsel
    gdf_additional_points.crs = gdf_wsel_wet_cells_cleaned_pnt.crs

    # Remove the duplicate points... that sit on two cells
    # Set the distance threshold
    threshold_distance = 0.01  # in feet

    # Merge points within the threshold distance
    gdf_additional_points_duplicates_removed = fn_merge_points_within_threshold(gdf_additional_points, threshold_distance, str_wsel_col_name)

    gdf_additional_points_duplicates_removed.crs = gdf_wsel_wet_cells_cleaned_pnt.crs

    points1 = gdf_wsel_wet_cells_cleaned_pnt[['geometry', str_wsel_col_name]].copy()

    if len(gdf_additional_points_duplicates_removed) > 0:
        points2 = gdf_additional_points_duplicates_removed[['geometry', str_wsel_col_name]].copy()
        points = pd.concat([points1, points2], axis=0)
    else:
        points = points1

    # create the points of computed wsel
    x = points.geometry.x.values
    y = points.geometry.y.values
    z = points[str_wsel_col_name].values

    # Define the extent of the cell point raster
    xmin, ymin, xmax, ymax = gdf_dissolved.total_bounds
    x_res = flt_pixel_width  # resolution in x direction
    y_res = flt_pixel_width  # resolution in y direction

    # Create the grid
    xi = np.arange(xmin, xmax, x_res)
    yi = np.arange(ymin, ymax, y_res)

    return PreparedInterpolationInputs(x, y, z, xi, yi, xmin, ymin, x_res, y_res)
# --------------------------------


# --------------------------------
def fn_create_raster_products(gdf_wsel_wet_cells_cleaned_pnt,
                              gdf_wsel_wet_cells_cleaned_ar,
                              str_wsel_col_name,
                              gdf_dissolved,
                              flt_pixel_width,
                              str_dem_output_folder,
                              geom,
                              str_dem_path):

    prep = fn_prepare_interpolation_inputs(gdf_wsel_wet_cells_cleaned_pnt,
                                           gdf_wsel_wet_cells_cleaned_ar,
                                           str_wsel_col_name,
                                           gdf_dissolved,
                                           flt_pixel_width)
    x, y, z = prep.x, prep.y, prep.z
    xmin, ymin, x_res, y_res = prep.xmin, prep.ymin, prep.x_res, prep.y_res

    xi, yi = np.meshgrid(prep.x_axis, prep.y_axis)

    # Perform bilinear interpolation
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Define the affine transformation
    transform = Affine.translation(xmin, ymin) * Affine.scale(x_res, y_res)

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # Begin the creation of the raster products
    list_raster_to_purge = []

    # ~~~~~~~
    # 01 - the wsel raster that is not masked

    # Define output raster file path
    output_raster_path_01 = os.path.join(str_dem_output_folder,'01_' + str_wsel_col_name + '_interp.tif')

    # Write the interpolated raster to a GeoTIFF file
    with rasterio.open(output_raster_path_01, 'w', driver='GTiff', height=zi.shape[0], width=zi.shape[1], count=1,
                       dtype=zi.dtype,
                       crs=gdf_wsel_wet_cells_cleaned_ar.crs,
                       transform=transform) as dst:
        dst.write(zi, 1)

    list_raster_to_purge.append(output_raster_path_01)
    
    # ~~~~~~~
    # 02 - the masked wsel

    # Open the raster dataset in read mode
    with rasterio.open(output_raster_path_01, 'r') as dst:

        # Mask the interpolated raster with the polygon
        zi_masked, transform_masked = rasterio.mask.mask(dst, [geom], crop=True, nodata=np.nan)  # Specify nodata value as np.nan

    # Define output masked raster file path
    output_masked_raster_path_02 = os.path.join(str_dem_output_folder,'02_' + str_wsel_col_name + '_interp_masked.tif')

    # Write the masked interpolated raster to a GeoTIFF file
    with rasterio.open(output_masked_raster_path_02, 'w', driver='GTiff', height=zi_masked.shape[1], width=zi_masked.shape[2], count=1,
                       dtype=zi_masked.dtype,
                       crs=gdf_wsel_wet_cells_cleaned_ar.crs,
                       transform=transform_masked,
                       nodata=np.nan) as dst_masked:  # Specify nodata value as np.nan
        dst_masked.write(zi_masked[0], 1)

    list_raster_to_purge.append(output_masked_raster_path_02)
    
    # ~~~~~~~
    # 03 and 04 - aligning the wsel and ground rasters

    try:
        # Open and read the first raster
        with rasterio.open(output_masked_raster_path_02) as src1:
            raster1 = src1.read(1)
            transform1 = src1.transform
            crs1 = src1.crs

            try:
                # Open and read the second raster (ground dem raster)
                with rasterio.open(str_dem_path) as src2:
                    raster2 = src2.read(1)
                    transform2 = src2.transform
                    crs2 = src2.crs

                    # TODO - I want all rasters to be alligned to the limits of the terrain.  This should
                    # preserve the same pixels on all the 'runs'. - 2024.07.08
                    
                    bbox = gpd.GeoDataFrame(geometry=[box(*src2.bounds), box(*src2.bounds)], crs=crs1)
                    common_extent = bbox.unary_union.bounds

                    # Resample and align rasters
                    dst_crs = crs1
                    transform, width, height = calculate_default_transform(crs2, dst_crs, src2.width, src2.height, *src2.bounds)
                    kwargs = src2.meta.copy()
                    kwargs.update({
                        'crs': dst_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    raster2_aligned = np.zeros_like(raster1)
                    reproject(
                        source=raster2,
                        destination=raster2_aligned,
                        src_transform=src2.transform,
                        src_crs=src2.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

                    # Clip rasters
                    output_raster_path_03 = os.path.join(str_dem_output_folder,'03_' + str_wsel_col_name + '_interp_mask_align.tif')
                    output_raster_path_04 = os.path.join(str_dem_output_folder,'04_' + str_wsel_col_name + '_ground_align.tif')

                    with rasterio.open(output_raster_path_03, 'w', **src1.meta) as dst1:
                        out_img1, out_transform1 = mask(src1, [bbox.iloc[0]['geometry']], crop=True)
                        dst1.write(out_img1)

                    with rasterio.open(output_raster_path_04, 'w', **src2.meta) as dst2:
                        out_img2, out_transform2 = mask(src2, [bbox.iloc[1]['geometry']], crop=True)
                        dst2.write(out_img2)

                    list_raster_to_purge.append(output_raster_path_03)
                    list_raster_to_purge.append(output_raster_path_04)

            except rasterio.errors.RasterioIOError as e:
                print("Error opening or processing raster file:", e)

    except rasterio.errors.RasterioIOError as e:
        print("Error opening raster file:", e)
        
    return list_raster_to_purge
# --------------------------------


# -------------------
def fn_raster_subtraction(raster1_path, raster2_path, output_folder, str_wsel_col_name):
    from osgeo import gdal, gdalconst
    
    list_created_rasters = []
    
    # Open the first raster
    raster1 = gdal.Open(raster1_path, gdalconst.GA_ReadOnly)
    if raster1 is None:
        raise FileNotFoundError(f"Raster not found at {raster1_path}")

    # Open the second raster
    raster2 = gdal.Open(raster2_path, gdalconst.GA_ReadOnly)
    if raster2 is None:
        raise FileNotFoundError(f"Raster not found at {raster2_path}")

    # Get the extent and resolution of the first raster
    x_min, pixel_width, _, y_max, _, pixel_height = raster1.GetGeoTransform()
    x_max = x_min + raster1.RasterXSize * pixel_width
    y_min = y_max + raster1.RasterYSize * pixel_height

    # Resample the second raster to match the extent and resolution of the first raster
    #resampled_raster2_path = r'E:\working\05_ground_resampled_align.tif'
    resampled_raster2_path = os.path.join(output_folder,'05_' + str_wsel_col_name + '_ground_resampled_align.tif')
    list_created_rasters.append(resampled_raster2_path)
    
    gdal.Warp(resampled_raster2_path, raster2, format='GTiff', outputBounds=[x_min, y_min, x_max, y_max], xRes=pixel_width, yRes=pixel_height)

    # Perform raster subtraction
    raster1_data = raster1.ReadAsArray()
    resampled_raster2 = gdal.Open(resampled_raster2_path, gdalconst.GA_ReadOnly)
    resampled_raster2_data = resampled_raster2.ReadAsArray()

    result_data = raster1_data - resampled_raster2_data

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    
    output_raster_path_06 = os.path.join(output_folder,'06_' + str_wsel_col_name + '_depth.tif')
    output_raster = driver.Create(output_raster_path_06, raster1.RasterXSize, raster1.RasterYSize, 1, gdalconst.GDT_Float32)

    # Set projection and transform
    output_raster.SetProjection(raster1.GetProjection())
    output_raster.SetGeoTransform(raster1.GetGeoTransform())

    # TODO - 2024.02.23 -- likely writing two raster bands
    # Write result data to output raster
    output_band = output_raster.GetRasterBand(1)
    output_band.WriteArray(result_data)

    # Close rasters
    output_band = None
    output_raster = None
    raster1 = None
    resampled_raster2 = None
    
    list_created_rasters.append(output_raster_path_06)
    
    return(list_created_rasters)
# -------------------


# --------------------
def fn_fix_depth_raster(output_raster_path_06, str_dem_output_folder, str_wsel_col_name):

    list_depth_raster_path = []
    
    # Build the raster file paths
    output_raster_path_07 = os.path.join(str_dem_output_folder,'07_' + str_wsel_col_name + '_depth_with_nan.tif')

    with rasterio.open(output_raster_path_06, 'r') as src:
        # Read raster data
        data = src.read(1)

        # Set values less than zero to NaN
        data[data < 0] = float('nan')

        # Copy metadata
        kwargs = src.meta.copy()

        # Update metadata for the new file
        kwargs.update(dtype=rasterio.float32, compress='lzw')

        # Write the modified data to a new GeoTIFF file
        with rasterio.open(output_raster_path_07, 'w', **kwargs) as dst:
            dst.write(data, 1)
            
        list_depth_raster_path.append(output_raster_path_07)
        
        return(list_depth_raster_path)
# --------------------


# -------------------
def fn_raster_addition_wsel(ground_raster_path, depth_raster_path, output_folder,str_wsel_col_name):
    from osgeo import gdal, gdalconst

    list_wsel_raster_path = []
    
    # Open the GeoTIFF rasters
    ground_raster = gdal.Open(ground_raster_path)
    depth_raster = gdal.Open(depth_raster_path)

    # Read raster data as NumPy arrays
    ground_data = ground_raster.ReadAsArray()
    depth_data = depth_raster.ReadAsArray()

    # Convert depth data to float, as NaN is a floating-point concept
    depth_data = depth_data.astype(float)

    # Mask NaN values in depth data
    depth_data[np.isnan(depth_data)] = np.nan

    # Add the two rasters where depth is not NaN
    result = np.where(~np.isnan(depth_data), ground_data + depth_data, np.nan)

    # Save the result to a new GeoTIFF
    
    output_path = os.path.join(output_folder, '08_' + str_wsel_col_name + '_wsel_with_nan.tif')
    
    driver = gdal.GetDriverByName("GTiff")
    #output_raster = driver.Create(output_path, ground_raster.RasterXSize, ground_raster.RasterYSize, 1, gdal.GDT_Float32)
    output_raster = driver.Create(output_path, ground_raster.RasterXSize, ground_raster.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    output_raster.SetProjection(ground_raster.GetProjection())
    output_raster.SetGeoTransform(ground_raster.GetGeoTransform())
    output_band = output_raster.GetRasterBand(1)
    output_band.WriteArray(result)
    output_band.FlushCache()

    # Close the rasters
    ground_raster = None
    depth_raster = None
    output_raster = None
    
    list_wsel_raster_path.append(output_path)
    
    return(list_wsel_raster_path)
# -------------------


# ----------------------
def fn_purge_rasters(list_raster_to_purge):
    for file_path in list_raster_to_purge:
        try:
            os.remove(file_path)
            #print(f"File '{file_path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{file_path}': {e}")
# ----------------------


# ------------------------------------
def fn_process_raster_stack(str_wsel_col_name,
                            gdf_cells_wsel,
                            gdf_center_points,
                            flt_pixel_width,
                            str_output_dir):
    
    geom, gdf_dissolved, gdf_wsel_wet_cells_cleaned_pnt, gdf_wsel_wet_cells_cleaned_ar = fn_wet_cells_per_run(str_wsel_col_name,
                                                                                                              gdf_cells_wsel,
                                                                                                              gdf_center_points)

    #print(f'Creating Rasters for {str_wsel_col_name}...')
    list_raster_to_purge = fn_create_raster_products(gdf_wsel_wet_cells_cleaned_pnt,
                                                     gdf_wsel_wet_cells_cleaned_ar,
                                                     str_wsel_col_name,
                                                     gdf_dissolved,
                                                     flt_pixel_width,
                                                     str_output_dir,
                                                     geom)

    print(f'Computing Depth and WSEL Raster for {str_wsel_col_name}...')
    list_created_rasters = fn_raster_subtraction(list_raster_to_purge[2],
                                                 list_raster_to_purge[3],
                                                 str_output_dir,
                                                 str_wsel_col_name)

    list_depth_raster_path = fn_fix_depth_raster(list_created_rasters[1],
                                                 str_output_dir,
                                                 str_wsel_col_name)

    list_wsel_raster_path = fn_raster_addition_wsel(list_created_rasters[0],
                                                    list_depth_raster_path[0],
                                                    str_output_dir,
                                                    str_wsel_col_name)

    # Purge the created rasters
    # Keeping #7(depth) & #8(wsel)
    fn_purge_rasters(list_raster_to_purge)
    fn_purge_rasters(list_created_rasters)
    
    return str_wsel_col_name
# ------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fn_make_floodplain_gdf(str_input_raster_filepath, flt_minvalue, flt_max_value):

    """
    Converts a raster file representing flooded areas to a GeoDataFrame of a single MultiPolygon representing the floodplain.

    Parameters:
    - str_input_raster_filepath (str): File path of the input raster file.
    - flt_minvalue (float): minimum value of depth allowed
    - flt_max_value (float): maximum value of depth allowed

    Returns:
    - gdf_single_floodplain (geopandas.GeoDataFrame): GeoDataFrame containing a merged MultiPolygon representing the floodplain.
    """
    
    # ----------------------
    # open the raster file and convert to 'binary' of flooded=1 and null = 255
    with rasterio.open(str_input_raster_filepath) as src:
        # Read the raster data as a numpy array
        data = src.read(1)

        # Set values greater than flt_minvalue and <= flt_max_value to 1, others to 255
        data = np.where((data > flt_minvalue) & (data <= flt_max_value), 1, 255).astype('uint8')

        # Create a new raster file with the updated data type and nodata value
        profile = src.profile
        profile.update(dtype=rasterio.uint8, nodata=255)

        # Use in-memory storage
        output_raster_memory = rasterio.MemoryFile()
        with output_raster_memory.open(**profile) as dst:
            dst.write(data, 1)

        #print("Conversion completed. Output raster stored in memory.")

    # ----------------------
    # convert the binary raster to geodataframe of polygons
    # Open the input GeoTIFF file
    with output_raster_memory.open() as src:
        # Read the raster data
        data = src.read(1, masked=True)  # using masked=True to handle NoData as a mask

        # Set all non-null values to 1
        data = np.where(data.mask, 255, 1).astype('uint8')

        # Extract shapes from the raster data
        shapes_gen = shapes(data, mask=data != 255, transform=src.transform)

        # Convert shapes to Shapely geometries and create a GeoDataFrame
        geometries = [shape(geom) for geom, _ in shapes_gen]
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=src.crs)

    # Merge all polygons into a single MultiPolygon
    merged_polygon = gdf['geometry'].unary_union

    # Create a new GeoDataFrame with the merged MultiPolygon
    gdf_single_floodplain = gpd.GeoDataFrame(geometry=[merged_polygon], crs=gdf.crs)

    # Close output_raster_memory to release the memory
    output_raster_memory.close()

    return(gdf_single_floodplain)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~


# -------------------------------
def fn_process_wsel(str_wsel_col_name, gdf_cells_wsel,
                    gdf_center_points, flt_pixel_width,
                    str_output_crs, flt_desired_res,
                    str_output_dir, str_dem_path):
    try:
        # Initialization
        #print('--------')
        #print(f'Processing {str_wsel_col_name}')
        
        # Wet points per run (wsel) and polygon boundary
        geom, gdf_dissolved, gdf_wsel_wet_cells_cleaned_pnt, gdf_wsel_wet_cells_cleaned_ar = fn_wet_cells_per_run(
            str_wsel_col_name, gdf_cells_wsel, gdf_center_points)
        
       # print('Creating Rasters...')
        list_raster_to_purge = fn_create_raster_products(
            gdf_wsel_wet_cells_cleaned_pnt, gdf_wsel_wet_cells_cleaned_ar, 
            str_wsel_col_name, gdf_dissolved, flt_pixel_width, str_output_dir, geom, str_dem_path)
        
        #print('Computing WSEL Raster...')
        list_created_rasters = fn_raster_subtraction(
            list_raster_to_purge[2], list_raster_to_purge[3], str_output_dir, str_wsel_col_name)
        
        list_depth_raster_path = fn_fix_depth_raster(
            list_created_rasters[1], str_output_dir, str_wsel_col_name)
        
        list_wsel_raster_path = fn_raster_addition_wsel(
            list_created_rasters[0], list_depth_raster_path[0], str_output_dir, str_wsel_col_name)
        
        # Nudge depth raster
        str_itermediate_raster_filepath = fn_set_nan(list_depth_raster_path[0])
        str_file_dir, str_file_name = os.path.split(str_itermediate_raster_filepath)
        str_output_filename = "09" + str_file_name[2:-18] + '_nudge.tif'
        str_output_path_depth = os.path.join(str_file_dir, str_output_filename)
        fn_nudge_raster(str_itermediate_raster_filepath, str_output_path_depth, str_output_crs, flt_desired_res)
        
        #print('Begin nudge wsel: 10...')
        # Nudge wsel raster
        str_itermediate_raster_filepath = fn_set_nan(list_wsel_raster_path[0])
        str_file_dir, str_file_name = os.path.split(str_itermediate_raster_filepath)
        str_output_filename = "10" + str_file_name[2:-18] + '_nudge.tif'
        str_output_path = os.path.join(str_file_dir, str_output_filename)
        fn_nudge_raster(str_itermediate_raster_filepath, str_output_path, str_output_crs, flt_desired_res)
        
        # Purge the created rasters
        list_created_rasters.append(list_depth_raster_path[0])
        list_created_rasters.append(list_wsel_raster_path[0])
        fn_purge_rasters(list_raster_to_purge)
        fn_purge_rasters(list_created_rasters)
        
        # Create a gdf of the floodplain from the depth raster
        flt_minvalue = 0
        flt_max_value = 10000
        gdf_floodplain = fn_make_floodplain_gdf(str_output_path_depth, flt_minvalue, flt_max_value)
        
        # Save the gdf_floodplain as a GeoPackage
        str_flood_limits_gpkg = os.path.join(str_output_dir, '11_' + str_wsel_col_name + '_floodplain_ar.gpkg')
        gdf_floodplain.to_file(str_flood_limits_gpkg)
        
        #print(f'Completed processing for {str_wsel_col_name}')
    except Exception as e:
        print(f'Error processing {str_wsel_col_name}: {e}')
# -------------------------------


# ----------------
def fn_str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', '1'}:
        return True
    elif value.lower() in {'false', 'f', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got '{value}'.")
# ----------------



# ---------------------------------------------
def fn_read_output_products(config):
    """[06_output_products] output CRS and resolution, required by AOI/rust."""
    if '06_output_products' not in config:
        raise ValueError('[06_output_products] section not found in the config file.')
    section = config['06_output_products']
    str_output_crs = section.get('str_output_crs', '')
    flt_desired_res = section.getfloat('flt_desired_res', 0)
    if not str_output_crs or not flt_desired_res > 0:
        raise ValueError('[06_output_products] needs str_output_crs and a positive flt_desired_res')
    return str_output_crs, flt_desired_res
# ---------------------------------------------


# ..................................
def fn_nudge_output_paths(str_output_dir, str_wsel_col_name):
    """Upstream names of products 09, 10 and 11."""
    return (os.path.join(str_output_dir, '09_' + str_wsel_col_name + '_depth_nudge.tif'),
            os.path.join(str_output_dir, '10_' + str_wsel_col_name + '_wsel_nudge.tif'),
            os.path.join(str_output_dir, '11_' + str_wsel_col_name + '_floodplain_ar.gpkg'))


def fn_remove_nudge_products(str_output_dir, str_wsel_col_name):
    for str_path in fn_nudge_output_paths(str_output_dir, str_wsel_col_name):
        if os.path.exists(str_path):
            os.remove(str_path)
# ..................................


# ----------------------------------
def fn_write_nudge_products(str_wsel_col_name, arr_depth, arr_wsel, transform, str_output_crs,
                            str_output_dir, b_aoi):
    """
    Write 09 (depth) and 10 (WSEL) as float32 LZW GeoTIFFs with NaN nodata,
    then 11 from 09 with the upstream fn_make_floodplain_gdf. Used for the
    rust engine and for AOI clips of either engine. Returns a status.
    """
    str_09, str_10, str_11 = fn_nudge_output_paths(str_output_dir, str_wsel_col_name)
    int_h, int_w = arr_depth.shape
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'width': int_w, 'height': int_h, 'count': 1,
               'crs': str_output_crs, 'transform': transform, 'nodata': np.nan,
               'compress': 'lzw', 'tiled': False, 'interleave': 'band'}
    for str_path, arr in ((str_09, arr_depth), (str_10, arr_wsel)):
        with rasterio.open(str_path, 'w', **profile) as dst:
            dst.write(np.asarray(arr, dtype=np.float32), 1)
    if os.path.exists(str_11):
        os.remove(str_11)

    flt_minvalue = 0
    flt_max_value = 10000
    if b_aoi and not np.any((arr_depth > flt_minvalue) & (arr_depth <= flt_max_value)):
        # a real AOI crop without flooded pixels: no floodplain polygon
        return 'no_flooded_pixels'
    gdf_floodplain = fn_make_floodplain_gdf(str_09, flt_minvalue, flt_max_value)
    gdf_floodplain.to_file(str_11)
    return 'complete'
# ----------------------------------


# ----------------------------------
def fn_postclip_python_products(str_wsel_col_name, aoi, str_output_crs, str_output_dir):
    """AOI for the python engine: clip the full-map 09/10, then rebuild 11."""
    str_09, str_10, _ = fn_nudge_output_paths(str_output_dir, str_wsel_col_name)
    if not (os.path.exists(str_09) and os.path.exists(str_10)):
        fn_remove_nudge_products(str_output_dir, str_wsel_col_name)
        return {'status': 'failed', 'error': 'python engine wrote no 09/10 (see printed error)'}
    with rasterio.open(str_09) as src:
        arr_depth = src.read(1)
        transform = src.transform
        tup_shape = (src.height, src.width)
    with rasterio.open(str_10) as src:
        arr_wsel = src.read(1)
        if src.transform != transform or (src.height, src.width) != tup_shape:
            raise RuntimeError(f'{str_wsel_col_name}: 09 and 10 grids differ')
    win = mapping_aoi.fn_aoi_window(aoi, transform, tup_shape[1], tup_shape[0], str_output_crs)
    if win.is_empty:
        fn_remove_nudge_products(str_output_dir, str_wsel_col_name)
        return {**win.describe(), 'status': mapping_aoi.AOI_STATUS_EMPTY}
    str_status = fn_write_nudge_products(str_wsel_col_name,
                                         mapping_aoi.fn_apply_aoi_window(arr_depth, win),
                                         mapping_aoi.fn_apply_aoi_window(arr_wsel, win),
                                         win.transform, str_output_crs, str_output_dir, True)
    return {**win.describe(), 'status': str_status}
# ----------------------------------


# ----------------------------------
def fn_prepare_global_samples(str_wsel_col_name, gdf_cells_wsel, gdf_center_points, flt_pixel_width):
    """
    Global (never AOI-cropped) preparation for one profile, from the same
    upstream functions the python engine uses. Returns (prep, geom).
    """
    arr_values = gdf_cells_wsel[str_wsel_col_name].to_numpy(dtype=np.float64)
    if np.isinf(arr_values).any():
        # upstream only drops NaN; infinite WSE would enter the interpolation
        raise rust_mapping.UnsupportedForRustEngine(
            f'{str_wsel_col_name}: infinite WSE values are not supported')
    if not np.isfinite(arr_values).any():
        raise ValueError(f'{str_wsel_col_name}: no wet cells')
    geom, gdf_dissolved, gdf_pnt, gdf_ar = fn_wet_cells_per_run(str_wsel_col_name,
                                                                gdf_cells_wsel,
                                                                gdf_center_points)
    prep = fn_prepare_interpolation_inputs(gdf_pnt, gdf_ar, str_wsel_col_name,
                                           gdf_dissolved, flt_pixel_width)
    return prep, geom
# ----------------------------------


# ..................................
def fn_write_mapping_manifest(str_output_dir, dict_manifest):
    _fn_write_mapping_json(os.path.join(str_output_dir, rust_mapping.STR_RESULT_MANIFEST), dict_manifest)


def _fn_write_mapping_json(path, value):
    temporary = path + '.partial'
    try:
        with open(temporary, 'w') as stream:
            json.dump(value, stream, indent=1, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.remove(temporary)


def _fn_mapping_products(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in os.listdir(folder)
            if name.startswith(('09_', '10_', '11_'))
            and name.endswith(('.tif', '.gpkg')) and os.path.isfile(os.path.join(folder, name))]


def fn_generate_depth_wsel_rasters(str_config_file_path, str_hdf_file_path, str_dem_path,
                                   str_hydraulic_results_gpkg, str_output_dir, b_print_output,
                                   *, mapping_engine=None, rust_binary=None, mapping_threads=None,
                                   bbox=None, polygon=None, polygon_layer=None, aoi_crs=None,
                                   **rust_options):
    """Map Step05 profiles; optional engines/AOIs require fresh mapping outputs.

    The unchanged Python default keeps its original output policy. Rust or AOI
    runs record in-progress/completed/failed status and reject existing mapping
    products or manifests, preventing an old map from becoming a new result.
    """
    config = configparser.ConfigParser()
    config.read(str_config_file_path)
    options = dict(mapping_engine=mapping_engine, rust_binary=rust_binary,
                   mapping_threads=mapping_threads, bbox=bbox, polygon=polygon,
                   polygon_layer=polygon_layer, aoi_crs=aoi_crs, **rust_options)
    opts = rust_mapping.fn_resolve_mapping_options(config, engine=mapping_engine,
        rust_binary=rust_binary, threads=mapping_threads, bbox=bbox, polygon=polygon,
        polygon_layer=polygon_layer, aoi_crs=aoi_crs, **rust_options)
    optional = opts.engine == rust_mapping.ENGINE_RUST or opts.has_aoi_options
    common = (str_config_file_path, str_hdf_file_path, str_dem_path,
              str_hydraulic_results_gpkg, str_output_dir, b_print_output)
    if not optional:
        return _fn_generate_depth_wsel_rasters(*common, **options)
    if opts.engine == rust_mapping.ENGINE_RUST:
        rust_mapping.fn_resolve_rust_binary(opts.rust_binary)
    output_crs, _ = fn_read_output_products(config)
    mapping_aoi.fn_resolve_aoi(bbox=opts.bbox, polygon=opts.polygon,
        polygon_layer=opts.polygon_layer, aoi_crs=opts.aoi_crs, default_bbox_crs=output_crs)
    manifest_path = os.path.join(str_output_dir, rust_mapping.STR_RESULT_MANIFEST)
    report_path = os.path.join(str_output_dir, rust_mapping.STR_ENGINE_REPORT)
    if _fn_mapping_products(str_output_dir) or any(os.path.exists(p) for p in (manifest_path, report_path)):
        raise FileExistsError('Optional mapping requires fresh outputs; use a new output directory.')
    os.makedirs(str_output_dir, exist_ok=True)
    marker = {'engine': opts.engine, 'status': 'in_progress', 'columns': {}}
    fn_write_mapping_manifest(str_output_dir, marker)
    if opts.engine == rust_mapping.ENGINE_RUST:
        _fn_write_mapping_json(report_path, marker)
    try:
        result = _fn_generate_depth_wsel_rasters(*common, **options)
    except Exception as error:
        try:
            with open(manifest_path) as stream:
                marker = json.load(stream)
        except (OSError, ValueError):
            marker = {'engine': opts.engine, 'columns': {}}
        # A finished per-profile manifest may retain good profiles. A session
        # abort has no such manifest: remove its uncommitted product set.
        status = 'failed' if marker.get('columns') else 'aborted'
        if status == 'aborted':
            for path in _fn_mapping_products(str_output_dir):
                os.remove(path)
        marker.update(status=status, error=f'{type(error).__name__}: {error}')
        fn_write_mapping_manifest(str_output_dir, marker)
        if opts.engine == rust_mapping.ENGINE_RUST:
            try:
                with open(report_path) as stream:
                    report = json.load(stream)
            except (OSError, ValueError):
                report = {}
            report.update(status=status, error=marker['error'])
            _fn_write_mapping_json(report_path, report)
        raise
    with open(manifest_path) as stream:
        marker = json.load(stream)
    marker['status'] = 'complete'
    fn_write_mapping_manifest(str_output_dir, marker)
    if opts.engine == rust_mapping.ENGINE_RUST:
        with open(report_path) as stream:
            report = json.load(stream)
        report['status'] = 'complete'
        _fn_write_mapping_json(report_path, report)
    return result
# ..................................


# ---------------------------------------------
def fn_generate_rasters_rust(str_hdf_file_path, str_dem_path, str_hydraulic_results_gpkg,
                             str_output_dir, opts, aoi, str_output_crs, flt_desired_res):
    """Step 05 with the rust engine: python prepares, ras2fim-batch maps."""
    str_binary = rust_mapping.fn_resolve_rust_binary(opts.rust_binary)
    int_dst_epsg = rust_mapping.fn_output_epsg(str_output_crs)

    gdf_cells_wsel = gpd.read_file(str_hydraulic_results_gpkg, layer='02_cells_wsel_ar')
    dict_terrain = rust_mapping.fn_terrain_identity(str_binary, str_dem_path, gdf_cells_wsel.crs,
                                                    opts.rust_terrain_crs_fnv1a64)
    list_unique_indices_sorted = sorted(gdf_cells_wsel['cell_idx'].unique())
    list_wsel_col_names = [col for col in gdf_cells_wsel.columns if col.startswith('wsel')]
    gdf_center_points = fn_gdf_cell_centerpoints(str_hdf_file_path,
                                                 list_unique_indices_sorted,
                                                 gdf_cells_wsel)
    if opts.engine == rust_mapping.ENGINE_RUST:
        with rasterio.open(str_dem_path) as terrain:
            flt_pixel_width = terrain.res[0]
    else:
        flt_pixel_width = fn_get_raster_resolution(str_dem_path)
    with rasterio.open(str_dem_path) as src:
        tup_terrain_bounds = tuple(src.bounds)

    os.makedirs(str_output_dir, exist_ok=True)
    dict_columns = {}
    list_reports = []
    str_work = tempfile.mkdtemp(prefix='.rust_step05_', dir=str_output_dir)
    try:
        for int_batch, list_batch in enumerate(rust_mapping.fn_chunks(list_wsel_col_names,
                                                                      opts.rust_profiles_per_request)):
            str_batch = os.path.join(str_work, f'batch_{int_batch:04d}')
            os.makedirs(os.path.join(str_batch, 'masks'))
            list_profiles = []
            for str_col in list_batch:
                try:
                    prep, geom = fn_prepare_global_samples(str_col, gdf_cells_wsel,
                                                           gdf_center_points, flt_pixel_width)
                    grids = rust_mapping.fn_derive_step05_grids(prep, geom, gdf_cells_wsel.crs,
                                                                tup_terrain_bounds, str_output_crs,
                                                                flt_desired_res)
                    win = None
                    str_mask = None
                    if aoi is not None:
                        win = mapping_aoi.fn_aoi_window(aoi, grids.dest_transform, grids.dest_width,
                                                        grids.dest_height, str_output_crs)
                        if win.is_empty:
                            fn_remove_nudge_products(str_output_dir, str_col)
                            dict_columns[str_col] = {**win.describe(),
                                                     'status': mapping_aoi.AOI_STATUS_EMPTY}
                            continue
                        if win.mask is not None:
                            str_mask = f'masks/{str_col}.u8'
                            np.ascontiguousarray(win.mask, dtype=np.uint8).tofile(
                                os.path.join(str_batch, str_mask))
                    str_inputs = f'inputs/{str_col}'
                    rust_mapping.fn_write_profile_inputs(os.path.join(str_batch, str_inputs),
                                                         str_col, prep, grids)
                    # the wet mask now lives in the prepared file; keep only grid metadata
                    grids.wet_mask = None
                    del prep, geom
                    list_profiles.append(rust_mapping.ProfileRequest(str_col, str_inputs, grids,
                                                                     win, str_mask))
                except Exception as e:
                    print(f'Error processing {str_col}: {e}')
                    fn_remove_nudge_products(str_output_dir, str_col)
                    dict_columns[str_col] = {'status': 'failed', 'error': f'{type(e).__name__}: {e}'}

            if list_profiles:
                list_results, dict_report = rust_mapping.fn_run_step05_request(
                    str_binary, str_batch, int_batch, dict_terrain, opts, list_profiles, int_dst_epsg)
                list_reports.append({'batch': int_batch, 'profiles': [p.column for p in list_profiles],
                                     'report': dict_report})
                dict_windows = {p.column: p.aoi_window for p in list_profiles}
                for result in list_results:
                    if result.status != 'complete':
                        print(f'Error processing {result.column}: {result.error}')
                        fn_remove_nudge_products(str_output_dir, result.column)
                        dict_columns[result.column] = {'status': 'failed', 'error': result.error}
                        continue
                    str_status = fn_write_nudge_products(result.column, result.depth, result.wse,
                                                         result.transform, str_output_crs,
                                                         str_output_dir, aoi is not None)
                    win = dict_windows[result.column]
                    dict_columns[result.column] = {**(win.describe() if win is not None else {}),
                                                   'status': str_status}
            if not opts.rust_keep_work:
                shutil.rmtree(str_batch)
    finally:
        if not opts.rust_keep_work:
            shutil.rmtree(str_work, ignore_errors=True)

    # Native engine reports (cache, source pixels, stage timings) for benchmarks
    with open(os.path.join(str_output_dir, rust_mapping.STR_ENGINE_REPORT), 'w') as f:
        json.dump({'schema': 'ras2fim-2d-rust-mapping-report/1',
                   'binary': os.path.basename(str_binary),
                   'terrain_epsg': dict_terrain['epsg'],
                   'crs_identity': dict_terrain['crs_identity'],
                   'batches': list_reports}, f, indent=1)
    return dict_columns, (str_work if opts.rust_keep_work else None)
# ---------------------------------------------


# ---------------------------------------------
def _fn_generate_depth_wsel_rasters(str_config_file_path,
                                   str_hdf_file_path,
                                   str_dem_path,
                                   str_hydraulic_results_gpkg,
                                   str_output_dir,
                                   b_print_output,
                                   *,
                                   mapping_engine=None,
                                   rust_binary=None,
                                   mapping_threads=None,
                                   bbox=None,
                                   polygon=None,
                                   polygon_layer=None,
                                   aoi_crs=None,
                                   **rust_options):
    """
    Step 05. Without options (and with an empty [05_raster_engine]) this is
    the original python engine, unchanged. Options (keyword > config):
      mapping_engine   'python' (default) or 'rust' (opt-in; needs the binary)
      rust_binary      path to ras2fim-batch (default: PATH lookup)
      mapping_threads  python pool processes / rust threads
      bbox             (xmin, ymin, xmax, ymax); CRS aoi_crs or the output CRS
      polygon          GeoJSON/GPKG path or Shapely (Multi)Polygon
      polygon_layer, aoi_crs
      rust_options     rust_max_cache_mib, rust_max_profile_mib,
                       rust_profiles_per_request, rust_max_inflight_profiles,
                       rust_nudge_method, rust_terrain_crs_fnv1a64, rust_keep_work
    Returns None for the unchanged default (no AOI, python engine);
    otherwise a dict of per-column results.
    """
    config_engine = configparser.ConfigParser()
    config_engine.read(str_config_file_path)
    opts = rust_mapping.fn_resolve_mapping_options(config_engine,
                                                   engine=mapping_engine,
                                                   rust_binary=rust_binary,
                                                   threads=mapping_threads,
                                                   bbox=bbox,
                                                   polygon=polygon,
                                                   polygon_layer=polygon_layer,
                                                   aoi_crs=aoi_crs,
                                                   **rust_options)
    aoi = None
    if opts.engine == rust_mapping.ENGINE_RUST or opts.has_aoi_options:
        str_aoi_output_crs, flt_aoi_desired_res = fn_read_output_products(config_engine)
        aoi = mapping_aoi.fn_resolve_aoi(bbox=opts.bbox, polygon=opts.polygon,
                                         polygon_layer=opts.polygon_layer, aoi_crs=opts.aoi_crs,
                                         default_bbox_crs=str_aoi_output_crs)

    # supress all warnings
    warnings.filterwarnings("ignore", category=UserWarning )

    if b_print_output:
        print(" ")
        print("+=================================================================+")
        print("|       GENERATE DEPTH AND WATER SURFACE ELEVATION RASTERS        |")
        print("|                Created by Andy Carter, PE of                    |")
        print("|             Center for Water and the Environment                |")
        print("|                 University of Texas at Austin                   |")
        print("+-----------------------------------------------------------------+")


        print("  ---(c) INPUT GLOBAL CONFIGUTATION FILE: " + str_config_file_path)
        print("  ---(i) INPUT HEC-RAS PLAN (HDF): " + str_hdf_file_path)
        print("  ---(t) INPUT TERRAIN (TIF): " + str_dem_path)
        print("  ---(g) INPUT HYDRAUNIC RESULTS (GPKG): " + str_hydraulic_results_gpkg)
        print("  ---(o) OUTPUT PATH: " + str_output_dir)
        print("  ---[r] PRINT OUTPUT: " + str(b_print_output))
        if not opts.is_legacy_default:
            print("  ---[e] MAPPING ENGINE: " + opts.engine)
            if aoi is not None:
                print("  ---[a] AREA OF INTEREST: " + aoi.source + " (" + aoi.crs.to_string() + ")")
        print("===================================================================")

    if opts.engine == rust_mapping.ENGINE_RUST:
        dict_columns, str_kept = fn_generate_rasters_rust(str_hdf_file_path, str_dem_path,
                                                          str_hydraulic_results_gpkg, str_output_dir,
                                                          opts, aoi, str_aoi_output_crs,
                                                          flt_aoi_desired_res)
        fn_write_mapping_manifest(str_output_dir, {
            'engine': opts.engine, 'aoi': aoi.describe() if aoi else None,
            'threads': opts.resolved_threads(), 'kept_work_dir': str_kept, 'columns': dict_columns})
        list_failed = [c for c, d in dict_columns.items() if d['status'] == 'failed']
        if list_failed:
            raise rust_mapping.RustMappingError(f'rust mapping failed for {len(list_failed)} profile(s): '
                                                f'{list_failed}; see {rust_mapping.STR_RESULT_MANIFEST}')
        return dict_columns

    # Read the GeoPackage file
    gdf_cells_wsel = gpd.read_file(str_hydraulic_results_gpkg, layer='02_cells_wsel_ar')

    # Get unique values of 'cell_idx' and sort them
    list_unique_indices_sorted = sorted(gdf_cells_wsel['cell_idx'].unique())

    list_wsel_col_names = [col for col in gdf_cells_wsel.columns if col.startswith('wsel')]

    # Create a gdf of all cells that are "wet"
    gdf_center_points = fn_gdf_cell_centerpoints(str_hdf_file_path,
                                                 list_unique_indices_sorted,
                                                 gdf_cells_wsel)

    # get the resolution of the ground DEM
    if opts.engine == rust_mapping.ENGINE_RUST:
        with rasterio.open(str_dem_path) as terrain:
            flt_pixel_width = terrain.res[0]
    else:
        flt_pixel_width = fn_get_raster_resolution(str_dem_path)

    # --- Read variables from config.ini ---
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the config.ini file
    config.read(str_config_file_path)

    # ----
    # Get the variables in the [04_time_gradient] section
    if '04_time_gradient' in config:
        section = config['06_output_products']

        # Read variables in the section
        str_output_crs = section.get('str_output_crs', '')
        flt_desired_res = section.getfloat('flt_desired_res', 0)
    else:
        print("[06_output_products] section not found in the config file.")
    # ----

    # **********
    # HARED CODED MAX NUMBER OF PROCESSORS
    int_max_processors = 16
    # **********
    num_processors = (mp.cpu_count() - 1)

    if num_processors > int_max_processors:
        num_processors = int_max_processors

    if opts.threads is not None:
        num_processors = opts.threads

    # TODO -- 2025.01.09 - add tqdm progress bar to this multiprocessing.
    # Create a pool of workers
    with mp.Pool(processes=num_processors) as pool:
        # Use partial to pass additional arguments to process_wsel
        from functools import partial
        process_func = partial(
            fn_process_wsel,
            gdf_cells_wsel=gdf_cells_wsel,
            gdf_center_points=gdf_center_points,
            flt_pixel_width=flt_pixel_width,
            str_output_crs=str_output_crs,
            flt_desired_res=flt_desired_res,
            str_output_dir=str_output_dir,
            str_dem_path=str_dem_path
        )
        pool.map(process_func, list_wsel_col_names)

    if aoi is None:
        return None

    # python engine AOI: clip each full map (the AOI baseline)
    dict_columns = {}
    for str_col in list_wsel_col_names:
        dict_columns[str_col] = fn_postclip_python_products(str_col, aoi, str_aoi_output_crs,
                                                            str_output_dir)
    fn_write_mapping_manifest(str_output_dir, {
        'engine': opts.engine, 'aoi': aoi.describe(), 'threads': num_processors, 'columns': dict_columns})
    failed = [name for name, info in dict_columns.items() if info['status'] == 'failed']
    if failed:
        raise rust_mapping.RustMappingError(f'python AOI mapping failed for {len(failed)} profile(s): {failed}')
    return dict_columns
# ---------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    flt_start_run = time.time()
    
    parser = argparse.ArgumentParser(description='======= GENERATE DEPTH AND WATER SURFACE ELEVATION RASTERS =======')
    
    parser.add_argument('-c',
                        dest = "str_config_file_path",
                        help=r'REQUIRED: Global configuration filepath Example: C:\Users\civil\ras2fim-2d\src\python_code\config_global.ini',
                        required=True,
                        default=r'C:\Users\civil\ras2fim-2d\src\python_code\config_global.ini',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-i',
                        dest = "str_hdf_file_path",
                        help=r'REQUIRED: path to plan HDF file (HDF) Example: E:\ras2fim_test_20250107\03_run_hdf\1884650_wb-2410416_wb-2410422_21-hr_7518-cfs_to_16893-cfs_step_500-cfs.p01.hdf',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-t',
                        dest = "str_dem_path",
                        help=r'REQUIRED: path to terrain dem (TIF) Example: E:\ras2fim_test_20250107\processing\02_model_copies\source_terrain',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-g',
                        dest = "str_hydraulic_results_gpkg",
                        help=r'REQUIRED: hydraulic results geopackage (geopackage) Example: E:\ras2fim_test_20250107\processing\05_large_terrain',
                        required=True,
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('-o',
                        dest = "str_output_dir",
                        help=r'REQUIRED: directory to write model raster output Example: E:\ras2fim_test_20250107\processing\05_large_terrain',
                        required=True,
                        metavar='DIR',
                        type=str)
    
    parser.add_argument('-r',
                        dest = "b_print_output",
                        help=r'OPTIONAL: Print output messages Default: True',
                        required=False,
                        default=True,
                        metavar='T/F',
                        type=fn_str_to_bool)
    
    rust_mapping.fn_add_mapping_arguments(parser)

    args = vars(parser.parse_args())
    
    str_config_file_path = args['str_config_file_path']
    str_hdf_file_path = args['str_hdf_file_path']
    str_dem_path = args['str_dem_path']
    str_hydraulic_results_gpkg = args['str_hydraulic_results_gpkg']
    str_output_dir = args['str_output_dir']
    b_print_output = args['b_print_output']

    fn_generate_depth_wsel_rasters(str_config_file_path,
                                   str_hdf_file_path,
                                   str_dem_path,
                                   str_hydraulic_results_gpkg,
                                   str_output_dir,
                                   b_print_output,
                                   **rust_mapping.fn_mapping_kwargs_from_args(args))

    flt_end_run = time.time()
    flt_time_pass = (flt_end_run - flt_start_run) // 1
    time_pass = datetime.timedelta(seconds=flt_time_pass)
    
    print('Compute Time: ' + str(time_pass))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
