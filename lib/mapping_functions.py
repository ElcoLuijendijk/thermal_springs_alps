import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as pl


import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.patches import Rectangle

from PIL import Image


def set_shade(a,intensity=None,cmap=pl.cm.jet,scale=10.0,azdeg=165.0,altdeg=45.0, vmin=None, vmax=None):
    
    ''' 
    
    sets shading for data array based on intensity layer
      or the data's value itself.
    inputs:
      a - a 2-d array or masked array
      intensity - a 2-d array of same size as a (no chack on that)
                        representing the intensity layer. if none is given
                        the data itself is used after getting the hillshade values
                        see hillshade for more details.
      cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
                  instance)
      scale,azdeg,altdeg - parameters for hilshade function see there for
                  more details
    output:
      rgb - an rgb set of the Pegtop soft light composition of the data and 
               intensity can be used as input for imshow()
    based on ImageMagick's Pegtop_light:
    http://www.imagemagick.org/Usage/compose/#pegtoplight
    
    source: http://rnovitsky.blogspot.de/2010/04/using-hillshade-image-as-intensity.html
    '''
    
    if vmin is None:
        vmin = a.min()
    if vmax is None:
        vmax = a.max()
    
    if intensity is None:
        # hilshading the data
        intensity = hillshade(a,scale=10.0,azdeg=165.0,altdeg=45.0)
    else:
        #or normalize the intensity
        intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    # get rgb of normalized data based on cmap
    rgb = cmap((a - vmin) / float(vmax - vmin))[:, :, :3]
    # form an rgb eqvivalent of intensity
    d = intensity.repeat(3).reshape(rgb.shape)
    # simulate illumination based on pegtop algorithm.
    rgb = 2 * d * rgb + (rgb**2) * (1 - 2 * d)
    
    return rgb


def hillshade(data, scale=10.0, azdeg=165.0, altdeg=45.0):

    ''' 
    convert data to hillshade based on matplotlib.colors.LightSource class.
    input:
         data - a 2-d array of data
         scale - scaling value of the data. higher number = lower gradient
         azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
                      270 west
         altdeg - where the light comes from: 0 horison ; 90 zenith
    output: a 2-d array of normalized hilshade
    '''
    # convert alt, az to radians
    az = azdeg*np.pi / 180.0
    alt = altdeg*np.pi / 180.0
    # gradient in x and y directions
    dx, dy = np.gradient(data / float(scale))
    slope = 0.5 * np.pi - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(dx, dy)
    intensity = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(-az - aspect - 0.5 * np.pi)
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())

    return intensity


def make_map_figure(proj, extent, panel_labels,
                    x_raster, y_raster, dem, 
                    x, y, z, vlim, class_int,
                    zlabel, hist_ylabel, legend_label, legend_label_nd, 
                    shps, shp_colors, shp_ls, shp_lws,
                    dem_min=-1000, dem_max=5000,
                    cmap=matplotlib.cm.get_cmap('coolwarm'),
                    long_ticks=[0.0, 5.0, 10.0, 15.0, 20.0], 
                    lat_ticks=[42.0, 44.0, 46.0, 48.0, 50.0],
                    fs_legend='small',
                    mark_highest_value=True,
                   n_highest=2,
                   add_histo=True):
    
    
    fig = pl.figure(figsize=(10, 6))
    panel = fig.add_subplot(1, 1, 1, projection=proj)
    
    xmin, xmax, ymin, ymax = extent
    
    panel.set_extent([xmin, xmax, ymin, ymax], crs=proj) 
    
    # calculate hillshade colors
    print('calculating hillshade')
    rgb = set_shade(dem, cmap=matplotlib.cm.Greys_r, vmin=dem_min, vmax=dem_max)

    # correct rgb array. not really sure why but otherwise the image isnt correct
    rgb = rgb[:, :-1, :]

    # convert rgb to tuple
    color_tuple = np.array([rii for ri in rgb for rii in ri])
    color_tuple = np.insert(color_tuple, 3, 1.0, axis=1)

    print('creating color image')
    #dem[dem <= 0 ] = np.nan
    im = panel.pcolormesh(x_raster, y_raster, dem, 
                          color=color_tuple, transform=proj, zorder=1)
    
    # plot additional shapefiles
    for shp, shp_color, shp_lsi, shp_lw in zip(shps, shp_colors, shp_ls, shp_lws):
        panel.add_geometries(shp.geometry, crs=proj, facecolor='none', edgecolor=shp_color, ls=shp_lsi, lw=shp_lw)
    
    vmin, vmax = vlim
    
    ind = np.isnan(z)
    leg_springs_nd = panel.scatter(x[ind], y[ind],
                                   vmin=vmin, vmax=vmax, zorder=102, s=40, marker='o',
                                   facecolor='None', edgecolor='black', transform=proj)

    ind = np.isnan(z) == False
    leg_springs = panel.scatter(x[ind], y[ind], c=z[ind],
                                vmin=vmin, vmax=vmax, zorder=103, s=40, marker='o',
                                cmap=cmap, facecolor='none', transform=proj, lw=1.5)
    
    leg_springs.set_facecolor('none')
    
    if mark_highest_value is True:
        #indh = np.argmax(z[ind])
        indh = z[ind].argsort()[-n_highest:][::-1]
        
        xs = x[ind][indh]
        ys = y[ind][indh]
        zs = z[ind][indh]
        
        leg_highest = panel.scatter(xs, ys, c=zs,
                                vmin=vmin, vmax=vmax, zorder=104, s=150, marker='*',
                                cmap=cmap, facecolor='none', transform=proj, lw=1.5)
        
        maxcol = cmap(1.0)
        
        if (len(xs) == 2) & (xs[1] == xs[0]) & (ys[1] == ys[0]):
            tekst = '1,2'
            panel.text(xs[0], ys[0] + 0.15, tekst, va='bottom', ha='center', weight='bold', transform=proj, fontsize='xx-large', zorder=105, color=maxcol)      
        else:
            for j, xi, yi in zip(itertools.count(), xs, ys):
                print('tekst: ', j, xi, yi + 0.1)
                panel.text(xi, yi + 0.15, '%s' % (j+1), va='bottom', ha='center', weight='bold', transform=proj, fontsize='xx-large', zorder=105, color=maxcol)

        print(xs, ys, [z[ind][indh]])
    
    #panel.coastlines('50m')
    #panel.add_feature(cfeature.OCEAN(scale='50m'), zorder=101)
    sea_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                            edgecolor='face',
                                            facecolor='lightblue', zorder=101)
    panel.add_feature(sea_50m)
    
    
    gl = panel.gridlines(xlocs=long_ticks, 
                         ylocs=lat_ticks,
                         crs=proj, zorder=2001, linestyle=':', 
                         draw_labels=True)
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_right = False

    panel.set_xlabel('Longitude')
    panel.set_ylabel('Latitude')
    
    legs = [leg_springs, leg_springs_nd]
    labels = [legend_label, legend_label_nd]
    
    #legend = panel.legend(legs, labels,fancybox=True, framealpha=0.75)
    #legend.legendPatch.set_facecolor('wheat')

    cb = fig.colorbar(leg_springs, shrink=0.3)
    cb.set_label(zlabel)
    
    ##################################
    # add inset panel with histogram:
    #################################
    if add_histo == True:
        inset_panel = inset_axes(panel,
                                 width="20%",  # width = 30% of parent_bbox
                                 height="25%",  # height : 1 inch
                                 loc="lower right", borderpad=3.5)

        inset_panel.tick_params(axis='both', which='major', labelsize=fs_legend)
        inset_panel.tick_params(axis='both', which='minor', labelsize=fs_legend)
        inset_panel.grid(b=False)

        bins = np.arange(vmin, vmax + class_int, class_int)
        freq, bins = np.histogram(z, bins=bins)

        bins_norm = (bins - vmin) / (vmax - vmin)

        bins_norm[bins_norm < 0.0] = 0.0
        bins_norm[bins_norm > 1.0] = 1.0
        color_vals = bins_norm

        #cmap = matplotlib.cm.get_cmap('GnBu')
        colors = cmap(color_vals[:-1])
        widths = bins[1:] - bins[:-1]
        inset_panel.bar(bins[:-1] + widths/2.0, freq,
                        width=widths, color=colors, edgecolor=colors, linewidth=0)

        #inset_panel.set_facecolor('lightgrey')
        #inset_panel.set_axis_bgcolor("lightgrey")
        #inset_panel.set_clip_on(False)

        #inset_panel.patch.set_alpha(0.75)
        #fig.tight_layout()
        inset_panel.set_xlabel(zlabel)
        inset_panel.set_ylabel(hist_ylabel)
        inset_panel.text(0.02, 1.03, panel_labels[1], weight='bold', 
                         fontsize=fs_legend, transform=inset_panel.transAxes)
        tekst = 'n=%i' % (len(z[np.isnan(z)==False]))
        inset_panel.text(0.98, 0.98, tekst, ha='right', va='top', transform=inset_panel.transAxes, fontsize=fs_legend)

        from matplotlib.patches import Rectangle
        bb = inset_panel.get_tightbbox(fig.canvas.get_renderer())
        #r = Rectangle(bb.bounds[:2], bb.bounds[2], bb.bounds[3], color='lightgrey', zorder=100)
        r = Rectangle((12.6, ymin), 7, 3.0, color='lightgrey', zorder=102, alpha=0.5)
        panel.add_patch(r)
    
    panel.text(0.00, 1.03, panel_labels[0], 
               fontsize=fs_legend,  weight='bold', transform=panel.transAxes)
    
    fig.tight_layout()
    
    return fig


def get_concat_v(im1, im2):
    
    """
    Merge two images
    """
    
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def crop_img(image):
    
    """
    Crop images, ie remove white space at edges
    """
    
    image_data = np.array(image)
    
    #ind = image_data[:, :] == np.array([255, 255, 255, 255])
    #image_data[ind] = 0 
    
    image_data_bw = image_data.min(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0)<255)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1)<255)[0]
    
    buffer = 10
    
    cropBox = (min(non_empty_rows) - buffer, max(non_empty_rows) + buffer, 
               min(non_empty_columns) - buffer, max(non_empty_columns) + buffer)

    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

    new_image = Image.fromarray(image_data_new)
    
    return new_image