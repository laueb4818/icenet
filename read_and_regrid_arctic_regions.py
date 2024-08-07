import numpy as np
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# set ellipsoid, coordinate system, grid metadata and polar stereo coordinates for
# NSIDC 25km polar stereographic sea ice data

# ellipsoid -- see http://nsidc.org/data/polar-stereo/ps_grids.html
a = 6378273.0
e = 0.081816153
b = a * ((1.0 - e * e) ** 0.5)
ellipsoid = iris.coord_systems.GeogCS(semi_major_axis=a, semi_minor_axis=b)

# coordinate system -- see ftp://sidads.colorado.edu/pub/tools/mapx/nsidc_maps/Nps.mpp
nps_cs = iris.coord_systems.Stereographic(
    90,
    -45,
    false_easting=0.0,
    false_northing=0.0,
    true_scale_lat=70,
    ellipsoid=ellipsoid,
)

# grid definition -- see ftp://sidads.colorado.edu/pub/tools/mapx/nsidc_maps/N3B.gpd
grid_length = 25000  # in m
nx = 304
ny = 448
cx = 153.5
cy = 233.5

# derive X and Y coordinates of pixel centres -- Y reversed so it starts at the bottom-left
x = np.linspace(-cx, (nx - 1) - cx, num=nx) * grid_length
y = np.linspace(cy - (ny - 1), cy, num=ny) * grid_length


# read region data
region_data = np.fromfile("region_n.msk", dtype="b", offset=300)
# region_data = np.fromfile(
#     "Arctic_region_mask_Meier_AnnGlaciol2007.msk", dtype=np.uint8
# ).reshape((448, 304))

# reshape and flip the data in the Y-direction
region_data = region_data.reshape((ny, nx))[::-1]

# convert to a cube
x_coord = iris.coords.DimCoord(
    x, "projection_x_coordinate", units="m", coord_system=nps_cs
)
y_coord = iris.coords.DimCoord(
    y, "projection_y_coordinate", units="m", coord_system=nps_cs
)
regions = iris.cube.Cube(region_data, dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)])


# plot the whole field
qplt.pcolormesh(regions)
plt.gca().coastlines()
plt.gca().gridlines()
plt.show()

# plot the field around Iceland
qplt.pcolormesh(regions)
plt.gca().set_extent((-26, -12, 63, 67), crs=ccrs.PlateCarree())
plt.gca().coastlines("50m")
plt.gca().gridlines()
plt.show()

# and Svalbard
qplt.pcolormesh(regions)
plt.gca().set_extent((8, 30, 76, 81), crs=ccrs.PlateCarree())
plt.gca().coastlines("50m")
plt.gca().gridlines()
plt.show()


# regrid to the EASE grid -- using nearest neighbour
sic = iris.load_cube(
    "ice_conc_nh_ease2-250_cdr-v2p0_197901021200.nc", "sea_ice_area_fraction"
)
for coord in ["projection_x_coordinate", "projection_y_coordinate"]:
    sic.coord(coord).convert_units("m")

regions_ease = regions.regrid(sic, iris.analysis.Nearest(extrapolation_mode="mask"))


# plot the regridded field
qplt.pcolormesh(regions_ease)
plt.gca().coastlines()
plt.gca().gridlines()
plt.show()

# plot the regridded field around Iceland
qplt.pcolormesh(regions_ease)
plt.gca().set_extent((-26, -12, 63, 67), crs=ccrs.PlateCarree())
plt.gca().coastlines("50m")
plt.gca().gridlines()
plt.show()

# and Svalbard
qplt.pcolormesh(regions_ease)
plt.gca().set_extent((8, 30, 76, 81), crs=ccrs.PlateCarree())
plt.gca().coastlines("50m")
plt.gca().gridlines()
plt.show()
