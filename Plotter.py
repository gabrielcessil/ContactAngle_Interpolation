import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib
import matplotlib as mpl
from PIL import Image
from skimage.measure import regionprops, label
import math

from utilities import LBPM_class_2_value

        
def Plot_Domain(values, filename, remove_value=[], colormap='cool', lbpm_class=True, special_colors={}):
    """
    Plot a 3D domain from a 3D NumPy array, highlighting:
        - Cells with value 0 as medium grey (0.5, 0.5, 0.5, 1.0)
        - Cells with value 1 as black (0.0, 0.0, 0.0, 1.0)
        - Other cells using a colormap

    Parameters:
        values (np.ndarray): 3D NumPy array of cell values.
        filename (str): Name of the output file (with path, without extension).
        remove_value (list): List of values to mark as ghost cells (optional).
        colormap (str): Colormap for non-zero and non-one values (default: 'cool').
        clim (list): Color range limits [min, max] for the colormap (default: [0, 255]).
    """

    # Ensure the folder for the output file exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Create structured grid (ImageData)
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Dimensions as points
    grid.origin = (0, 0, 0)  # Origin of the grid
    grid.spacing = (1, 1, 1)  # Uniform spacing

    # Assign cell values
    grid.cell_data["values"] = values.flatten(order="F")
    
    # Convert to an unstructured grid for filtering
    mesh = grid.cast_to_unstructured_grid()

    # Remove unwanted cells: OK
    for removed_value in remove_value:
        to_remove_mask = np.argwhere(mesh["values"] == removed_value)
        mesh.remove_cells(to_remove_mask.flatten(), inplace=True)

    # Separate different cell types
    special_cells = {}
    for value, color in special_colors.items():
        special_cells[value] = (mesh.extract_cells(np.where(mesh["values"] == value)[0]), color)

    
    # After locating fluid and solid, the values can be converted back to LBPM class
    other_cells_mask = np.where((mesh["values"] != 0) & (mesh["values"] != 1))[0]
    if lbpm_class: grid.cell_data["values"] = LBPM_class_2_value(grid.cell_data["values"])
    other_cells = mesh.extract_cells(other_cells_mask) 
    
    # Configure the plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)

    # Plot other cells with the colormap
    if other_cells.n_cells > 0:
        plotter.add_mesh(
            other_cells,
            cmap=colormap,
            #clim=clim,
            show_edges=False,
            lighting=True,
            smooth_shading=False,
            split_sharp_edges=False,
            scalar_bar_args={
                "title": "Continuous Range",
                "vertical": True,
                "title_font_size": 40,
                "label_font_size": 25,
                "position_x": 0.8,
                "position_y": 0.05,
                "height": 0.9,
                "width": 0.05,
                "n_labels": 10,
            },
        )
        
    for value, (cells, color) in special_cells.items():
        if cells.n_cells > 0:
            plotter.add_mesh(cells, color=color, show_scalar_bar=False)

    # Add axis indicators
    plotter.add_axes(line_width=5, cone_radius=0.6, shaft_length=0.9, tip_length=0.2, ambient=0.5, label_size=(0.25, 0.15))

    # Show grid bounds with labels
    plotter.show_bounds(
        grid='back', location='outer', ticks='both',
        show_xlabels=True, show_ylabels=True, show_zlabels=True,
        n_xlabels=4, n_ylabels=4, n_zlabels=4,
        font_size=15, xtitle='x', ytitle='y', ztitle='z'
    )

    # Show the visualization
    plotter.show(auto_close=False)

    # Save the visualization as an image
    plotter.screenshot(filename + ".png")
    plotter.save_graphic( filename + ".svg", raster=True, painter=True)
    plotter.close()

    return filename + ".png"

def Plot_Classified_Domain(values, filename, remove_value=[], labels={}, colormap='cool', 
                           show_label=True, special_colors={}, 
                           show_edges=False, lighting=True,
                           smooth_shading=False, split_sharp_edges=False,
                           ambient=None, diffuse=None, specular=None):
    # Ensure the output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # CREATE GRID STRUCTURE
    grid            = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Adjust for point-based representation
    grid.origin     = (0, 0, 0)  # Set grid origin
    grid.spacing    = (1, 1, 1)  # Uniform voxel spacing

    
    # Create mapping (normalization, treating values in any range as classes)
    unique_classes                  = np.unique(values)     # Ex: 3
    unique_classes_afterRemoving    = unique_classes[~np.isin(unique_classes, remove_value)]
    expoente        = math.floor(  math.log10((1/ (len(unique_classes_afterRemoving)-1)))   ) # Ex: -1
    delta           = 10 ** expoente        # Ex: 10^-1 = 0.1 (added for safety)
    n_decimals      = np.abs(expoente)+1    # Rounding decimals (for safety, one above the needed / one extra decimal )
    discrete_linspace_values    = np.linspace(start=0, stop=1, num=len(unique_classes_afterRemoving)) # Create a linear mapping with N value between 0 and 1
    normalized_mapping          = {class_i:     round(norm_val, n_decimals)  for class_i, norm_val in zip(unique_classes_afterRemoving, discrete_linspace_values)}
    denormalization_mapping     = {v:           round(k,        n_decimals)    for k, v in normalized_mapping.items()}
    
    # ASSING VALUES TO MESH STRUCTURE
    mesh_normalized_values          = np.vectorize(lambda x: normalized_mapping.get(x, np.nan))(values) # Normalize all the provided array    
    grid.cell_data["class_values"]  = mesh_normalized_values.flatten(order="F")    # Assign to grid
    mesh                            = grid.cast_to_unstructured_grid()             # Assign to mesh

    # REMOVE UNWATED CELLS
    nan_mask = np.isnan(mesh["class_values"])
    nan_cell_indices = np.where(nan_mask)[0]  # Flattened array of indices
    mesh.remove_cells(nan_cell_indices, inplace=True)
    for removed_value in remove_value:
        if removed_value in normalized_mapping:
            normalized_value_to_remove = normalized_mapping[removed_value]
            to_remove_mask = np.argwhere(mesh["class_values"] == normalized_value_to_remove)
            mesh.remove_cells(to_remove_mask.flatten(), inplace=True)
            
    # Unique values in mesh: after removing unwanted
    mesh_unique_values      = np.unique(mesh["class_values"])
    num_mesh_unique_values  = len(mesh_unique_values)
    
    # MANAGE COLORS
    # Get colormap for custom labels
    default_class_colors = {}
    for special_value, special_color in special_colors.items():
        if special_value in normalized_mapping:
            default_class_colors[ normalized_mapping[special_value] ] = special_color
    # Get colormap from values remaining in mesh
    nonDefault_classes = []
    for value in mesh_unique_values:
        if not (value in default_class_colors.keys()):
            nonDefault_classes.append(value)
    num_nonDefault_classes = len(nonDefault_classes)
    # Make colormap with custom colors
    if num_nonDefault_classes>0:
        colormap = mpl.colormaps[colormap].resampled(num_nonDefault_classes)
        color_space = np.linspace(1, 0, num_nonDefault_classes+1)[0:-1]
        if num_nonDefault_classes>1:
            color_space = color_space + (color_space[1]-color_space[0])/2
        generated_colors = colormap(color_space)
    color_list = []
    color_index = 0 # Avoid extreme colors (ex:too dark)  
    for val in mesh_unique_values:
        if val in default_class_colors:  # Use predefined colors for 0 and 1
            color = default_class_colors[val]
            color_list.append(mcolors.to_hex(color, keep_alpha=True))
        else:
            color = generated_colors[color_index] # Get color from colormap
            color_list.append(mcolors.to_hex(color, keep_alpha=True))
            color_index +=1
    
    # MAKE CUSTOM TICKS
    # Evenly distribute tick locations to avoid overlap, even if values are far apart
    if num_nonDefault_classes > 1:
        tick_positions = np.linspace(0, 1, num_mesh_unique_values+1)[0:-1]
        tick_positions = tick_positions + (tick_positions[1]-tick_positions[0])/2 # Set position to the middle of the range
    else:
        tick_positions = [discrete_linspace_values[0]]
    # Create annotation mapping with proper spacing
    annotations = {}
    
    for tick, val in zip(tick_positions, mesh_unique_values):
        original_val = denormalization_mapping[val]
        if int(denormalization_mapping[val]) in labels or float(denormalization_mapping[val]) in labels:
            annotations[tick] = f"                                     {labels[original_val]}" 
        else:
            annotations[tick] = f"                                     {original_val}"
    
    
    # MAKE THE PLOT
    plotter = pv.Plotter(window_size=[1920*3, 1080*3])  # Full HD resolution
    plotter.set_background(color=None)
    if mesh.n_cells > 0:
        plotter.add_mesh(
            mesh,
            scalars     =mesh["class_values"],
            show_scalar_bar=False,
            categories  =True,
            cmap        =color_list,
            show_edges  =show_edges,
            lighting    =lighting,
            smooth_shading      =smooth_shading,
            split_sharp_edges   =split_sharp_edges,
            ambient=ambient, 
            diffuse=diffuse, 
            specular=specular,
            #annotations         =annotations,
            #scalar_bar_args={
            #    "title": "    Classes",
            #    "vertical":         True,
            #    "title_font_size":  40,
            #    "label_font_size":  25,
            #    "position_x":       0.8,
            #    "position_y":       0.05,
            #    "height":           0.9,
            #    "width":            0.05,
            #    "n_labels":         0,
            #},"""
            clim=[0,1]  
        )
        
    """
    # Add axis in, dicators
    plotter.add_axes(
        line_width=5,      
        cone_radius=0.6,     
        shaft_length=0.9, 
        tip_length=0.2,
        ambient=0.5,
        label_size=(0.25, 0.15)
    )

    # Show grid bounds
    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        font_size=15,
        xtitle='x',
        ytitle='y',
        ztitle='z'
    )
    """

    # Show the visualization
    plotter.show(auto_close=False)

    # Save the visualization as an image
    plotter.screenshot(filename + ".png",  transparent_background=True)
    plotter.save_graphic( filename + ".svg", raster=False, painter=False)
    plotter.close()
    
    return filename + ".png"


from matplotlib.patches import Patch

def Plot_Classified_Domain_2D(values, filename, remove_value=[], labels={}, colormap='cool', show_label=True, special_colors={}):
    # Ensure the output directory exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # If input is 3D (1, H, W), take first slice
    if len(values.shape) == 3 and values.shape[0] == 1:
        values = values[0]

    # Get unique classes (ignoring remove_value)
    unique_classes = np.unique(values)
    unique_classes = unique_classes[~np.isin(unique_classes, remove_value)]

    expoente = math.floor(math.log10((1/(len(unique_classes)-1)))) if len(unique_classes) > 1 else 0
    delta = 10 ** expoente if len(unique_classes) > 1 else 1
    n_decimals = np.abs(expoente)+1 if len(unique_classes) > 1 else 1
    discrete_linspace_values = np.linspace(start=0, stop=1, num=len(unique_classes))
    normalized_mapping = {class_i: round(norm_val, n_decimals) for class_i, norm_val in zip(unique_classes, discrete_linspace_values)}
    denormalization_mapping = {v: round(k, n_decimals) for k, v in normalized_mapping.items()}

    # Normalize values
    normalized_values = np.vectorize(lambda x: normalized_mapping.get(x, np.nan))(values)

    # Handle special colors
    default_class_colors = {}
    for special_value, special_color in special_colors.items():
        if special_value in normalized_mapping:
            default_class_colors[normalized_mapping[special_value]] = special_color

    # Generate colors for remaining classes
    nonDefault_classes = [val for val in normalized_mapping.values() if val not in default_class_colors]
    num_nonDefault_classes = len(nonDefault_classes)
    if num_nonDefault_classes > 0:
        cmap = mpl.colormaps[colormap].resampled(num_nonDefault_classes)
        color_space = np.linspace(1, 0, num_nonDefault_classes+1)[0:-1]
        if num_nonDefault_classes > 1:
            color_space = color_space + (color_space[1]-color_space[0])/2
        generated_colors = cmap(color_space)

    # Build final color list
    color_list = []
    legend_elements = []
    color_index = 0
    for val in sorted(normalized_mapping.values()):
        if val in default_class_colors:
            color = default_class_colors[val]
        else:
            color = generated_colors[color_index]
            color_index += 1
        color_list.append(color)

        if show_label:
            original_val = denormalization_mapping[val]
            label_text = labels.get(original_val, str(original_val))
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label_text))

    # Create custom colormap
    cmap = mcolors.ListedColormap(color_list)
    bounds = np.linspace(0, 1, len(color_list)+1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(normalized_values, cmap=cmap, norm=norm, interpolation='none')

    # Add legend instead of colorbar
    if show_label:
        ax.legend(handles=legend_elements, loc='upper left', fontsize='xx-large', frameon=True)


    ax.set_xticks([])           # Hide x-axis ticks
    ax.set_yticks([])           # Hide y-axis ticks
    ax.set_title("")            # Remove title
    ax.set_xlabel("")           # Remove x-axis label
    ax.set_ylabel("")           # Remove y-axis label
    
    fig.tight_layout()
    


    plt.savefig(filename + ".png", bbox_inches='tight', dpi=300)
    plt.close()
    return filename + ".png"


def plot_hist(data, bins=30, title='Histogram', filename=None, notable=[], xlim=(), ylim=(), color='blue'):
    
    num_main_categories = len(data)
    
    fig, axes = plt.subplots(num_main_categories, 1, figsize=(15, 5 * num_main_categories))
    if num_main_categories == 1:
        axes = [axes]
    
    plt.subplots_adjust(top=0.9)  # Adjust layout to prevent title overlap
    plt.suptitle(title, fontsize=16, y=0.98)  # Move the title slightly upwards
    
    
    # Iterate over data_dict and check each element type
    for i, (category, content) in enumerate(data.items()):
        
        # Single histogram
        axes[i].hist(content, bins=bins, edgecolor='black', alpha=0.5, color=color, density=True, label=category)
        
        axes[i].set_title(f'{category}')
        for x_value in notable:
            axes[i].axvline(x_value, color='k', linestyle='dashed', linewidth=1, label='Notable' if 'Notable' not in axes[i].get_legend_handles_labels()[1] else None)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        if xlim: axes[i].set_xlim(xlim)
        if ylim: axes[i].set_ylim(ylim)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save or show the plot
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
    else:
        plt.show()


    
def plot_heatmap(array_2d, output_file="heatmap_hd", dpi=300, cmap="inferno", xlabel="X-axis", ylabel="Y-axis", title="Heatmap", colorbar_label="Value", vmin=None, vmax=None, grid=False):

    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    # If the first or last dimension is 1, squeeze it
    if array_2d.ndim == 3:
        if array_2d.shape[0] == 1:  # Case: (1, H, W) -> (H, W)
            array_2d = array_2d.squeeze(0)
        elif array_2d.shape[-1] == 1:  # Case: (H, W, 1) -> (H, W)
            array_2d = array_2d.squeeze(-1)
        
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 9))  # Full HD aspect ratio

    # Plot the heatmap with no interpolation (solid colors) and custom color range
    if vmin is None and vmax is None:
        heatmap = ax.imshow(array_2d, cmap=cmap, aspect="auto",
                        origin="upper", interpolation="none")
    else:
        heatmap = ax.imshow(array_2d, cmap=cmap, aspect="auto",
                        origin="upper", interpolation="none", vmin=vmin, vmax=vmax)

    # Add gridlines for the grid view
    if grid:
        ax.set_xticks(np.arange(-0.5, array_2d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, array_2d.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(colorbar_label)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save the plot in HD resolution
    plt.savefig(output_file+".png", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_classified_map(array_2d, output_file="classified_map", dpi=300, xlabel="X-axis", ylabel="Y-axis",
                        title="Classified Map", colorbar_label="Classes", grid=False, colormap='cool'):
    """
    Create and save a classified color map from a 2D NumPy array with distinct solid colors per class.

    Parameters:
        array_2d (np.ndarray): 2D NumPy array containing classified values.
        output_file (str): File name to save the classified map.
        dpi (int): Resolution of the output image.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        colorbar_label (str): Label for the colorbar.
        grid (bool): Whether to show a grid overlay.
    """
    
    # Ensure output directory exists
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Ensure the array is 2D
    if array_2d.ndim == 3:
        if array_2d.shape[0] == 1:
            array_2d = array_2d.squeeze(0)
        elif array_2d.shape[-1] == 1:
            array_2d = array_2d.squeeze(-1)

    # Get unique class values
    unique_values = np.unique(array_2d)
    unique_values = np.array(sorted(unique_values, key=lambda x: (x not in [0, 1], x)))

    non_default_values = unique_values[~np.isin(unique_values, [0, 1])]
    num_classes = len(non_default_values)
    
    if num_classes > 0:
        cmap = plt.cm.get_cmap(colormap, num_classes)
        colors = cmap(np.linspace(0, 1, num_classes))
    else:
        colors = []  # or set to None or default colors
        
        
    # Extract the colors
    generated_colors = list(colors)  # Convert to list to allow modifications
    
    # Define Grey for 0 and Black for 1
    default_class_colors = []
    
    if np.isin(unique_values, [0]).any():  
        default_class_colors.append((0.5, 0.5, 0.5, 1.0))  # Grey (for label 0)
    
    if np.isin(unique_values, [1]).any():  
        default_class_colors.append((0.0, 0.0, 0.0, 1.0))  # Black (for label 1)
        
    # Append the generated colors
    final_colors = default_class_colors + generated_colors  # Prepend grey and black
    
    listed_cmap = mcolors.ListedColormap(final_colors[:len(unique_values)])  # Ensure only the needed colors are used

    # Create a dictionary mapping class values to colorbar labels
    class_labels = {v: i for i, v in enumerate(unique_values)}

    # Replace array values with corresponding indices for correct color mapping
    indexed_array = np.vectorize(class_labels.get)(array_2d)

    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the classified data with discrete colors
    heatmap = ax.imshow(indexed_array, cmap=listed_cmap, aspect="auto",
                        origin="upper", interpolation="none")

    # Add gridlines if needed
    if grid:
        ax.set_xticks(np.arange(-0.5, array_2d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, array_2d.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Create a discrete colorbar
    cbar = plt.colorbar(heatmap, ax=ax, ticks=range(len(unique_values)))
    cbar.set_label(colorbar_label)
    cbar.set_ticks(range(len(unique_values)))
    cbar.set_ticklabels(unique_values)  # Ensure correct labels match colors

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save and display the plot
    plt.savefig(output_file + ".png", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()
    
    return output_file + ".png"


def create_gif_from_filenames(image_filenames, gif_filename, duration=200, loop=0, erase_plots=True):
    """
    Creates a GIF from a list of image filenames.

    Parameters:
        image_filenames (list): A list of filenames for the images to include in the GIF.
        gif_filename (str): Name of the output GIF file.
        duration (int): Duration of each frame in milliseconds.
        loop (int): Number of times to loop the GIF (0 for infinite).
        erase_plots (bool): If True, delete the individual image files after creating the GIF.
    """
    if not image_filenames:
        print("Error: No image filenames provided.")
        return

    images = [Image.open(file) for file in image_filenames]

    if not images:
        print("Error: Could not open any images from the provided filenames.")
        return
    images[0].save(gif_filename+".gif", save_all=True, append_images=images[1:], loop=loop, duration=duration)

    if erase_plots:
        for file in image_filenames:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")
    


def plot_labeled_clusters(cluster_image, labels, output_file="labeled_clusters"):
    """
    Overlay numerical labels on top of a colorized cluster image.

    Args:
        cluster_image (np.ndarray): 3D RGB image displaying clusters.
        labels (np.ndarray): 2D labeled segmentation array.
        output_file (str): Path to save the result.
    """
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        
    # Ensure the array is 2D
    if cluster_image.ndim == 3:
        if cluster_image.shape[0] == 1:
            cluster_image = cluster_image.squeeze(0)
        elif cluster_image.shape[-1] == 1:
            cluster_image = cluster_image.squeeze(-1)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the cluster image
    ax.imshow(cluster_image, interpolation="nearest")
    
    # Extract region properties
    props = regionprops(label(labels))  # Get properties of labeled clusters

    # Overlay text labels at cluster centroids
    for region in props:
        centroid = region.centroid  # (y, x) coordinates
        ax.text(centroid[1], centroid[0], f"{region.label}", 
                color="white", fontsize=12, fontweight="bold", 
                ha="center", va="center", bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title and save
    ax.set_title("Labeled Clusters")
    plt.savefig(output_file+".png", dpi=300, bbox_inches="tight")
    plt.show()
    
import matplotlib.ticker as ticker
def plot_mean_deviation(x, y_means, y_devs, title="Algorithm Performance",
                        xlabel="Samples Cells / Rock Surface Cells", ylabel="Accuracy",
                        filename="performance_plot",
                        dashed =None): # Removed .png extension as it will be added dynamically
    """
    Plots the mean accuracy and its deviation as shaded area.

    Parameters:
    - x: List or numpy array of x-axis values.
    - y_means: List or numpy array of mean accuracy values (y-axis).
    - y_devs: List or numpy array of standard deviation values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - filename: Base filename to save the plot (without extension).
    """
    if isinstance(y_means, list):
        y_means = np.array(y_means)
    if isinstance(y_devs, list):
        y_devs = np.array(y_devs)
    if isinstance(x, list):
        x = np.array(x)
        
    
    # Use a modern style
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] =  "DejaVu Serif"

    # Create figure with the specified window size (1920x1080 pixels)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 = 19.2 x 10.8 inches at 100 dpi
    
    if dashed is not None:
        ax.plot(dashed[0], dashed[1], '--', color='black', linewidth=2.4, alpha=0.6, label="Regressed")
        
    # Plot mean curve
    ax.plot(x, y_means, 'o-', color='#1f77b4', linewidth=3, label="Mean")

    # Fill shaded deviation
    ax.fill_between(x, y_means - y_devs, y_means + y_devs, color='#1f77b4', alpha=0.2, label="Deviation")
    
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=34) # Increased label font size
    ax.set_ylabel(ylabel, fontsize=34) # Increased label font size
    ax.set_title(title, fontsize=34, fontweight='bold') # Increased title font size

    # Customizing ticks
    ax.tick_params(axis='both', which='major', labelsize=28) # Increased tick label font size

    # Add more ticks for x-axis if x has enough distinct values
    # This will create ticks at each x-value provided, making it more detailed
    if len(x) > 1: ax.set_xticks(x)

    # Add a legend
    ax.legend(fontsize=32, loc="lower right", frameon=True, shadow=True) # Increased legend font size

    # Show grid for clarity
    # Changed grid color to a dark gray for better visibility
    ax.minorticks_on()

    # Apply grid to both major and minor ticks
    ax.grid(which='both', linestyle='--', alpha=0.5, color='black')  # adjust to taste

    # Save the figure in high resolution as SVG
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # integers
    
    

    plt.savefig(filename + ".svg", dpi=300, bbox_inches="tight") # Changed extension to .svg
    plt.show()
    plt.close(fig) # Close the figure to free up memory
    
    
def plot_multiple_mean_deviation(curve_dict,
                                  title="Algorithm Performance",
                                  xlabel="Samples Cells / Rock Surface Cells",
                                  ylabel="Accuracy",
                                  filename="performance_plot"):
    """
    Plots multiple mean curves with shaded deviation areas.

    Parameters:
    - curve_dict: Dict where each key is a label (e.g., "Curve 1") and each value is a tuple (x, y_mean, y_dev)
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    - filename: Base filename to save the plot (without extension)
    """

    # Use a modern style and font
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "DejaVu Serif"

    # Create figure with the specified window size (1920x1080 pixels)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)

    # Plot each curve with shaded deviation
    for label, (x, y_mean, y_dev) in curve_dict.items():
        x = np.array(x)
        y_mean = np.array(y_mean)
        y_dev = np.array(y_dev)

        ax.plot(x, y_mean, linewidth=3, marker='o', label=label+": mean")
        ax.fill_between(x, y_mean - y_dev, y_mean + y_dev, alpha=0.2, label=label+": std")

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=34)
    ax.set_ylabel(ylabel, fontsize=34)
    ax.set_title(title, fontsize=34, fontweight='bold')

    # Customizing ticks
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.minorticks_on()
    ax.grid(which='both', linestyle='--', alpha=0.5, color='black')

    # Set log scale and formatter
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Set consistent x-ticks using first curve's x-values (assumes shared domain)
    #shared_x = list(curve_dict.values())[0][0]
    #if len(shared_x) > 1:
    #    ax.set_xticks(shared_x)

    # Add a legend
    ax.legend(fontsize=32, loc="lower right", frameon=True, shadow=True)

    # Save figure
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(filename + ".svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    

