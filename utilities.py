import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter

def is_LBPM_class(value): return value > 73 #value < 0
    
# From LBPM-class to angle
def LBPM_class_2_value(array_or_scalar, keep_values=(1,0)):  # Keep_values as tuple for consistency
    if isinstance(array_or_scalar, (int, float, np.integer, np.floating)):  # Check for scalar types
        if array_or_scalar not in keep_values:
            return int(np.ceil(array_or_scalar)) - 74 #- int(np.ceil(array_or_scalar))  # Apply to scalar
        else:
            return array_or_scalar # Return scalar
    elif isinstance(array_or_scalar, np.ndarray):  # Check for NumPy array
        result = array_or_scalar.copy()
        mask = ~np.isin(array_or_scalar, keep_values)
        result[mask] = np.ceil(array_or_scalar[mask]).astype(int) - 74 #- np.ceil(array_or_scalar[mask]).astype(int)
        return result
    else:
        raise TypeError("Input must be a scalar (int, float) or a NumPy array.")


def value_2_LBPM_class(array_or_scalar, keep_values=(1,0)):
    if isinstance(array_or_scalar, (int, float, np.integer, np.floating)):
        if array_or_scalar not in keep_values:
            return int(np.ceil(array_or_scalar)) +74 #-int(np.ceil(array_or_scalar))
        else:
            return array_or_scalar
    elif isinstance(array_or_scalar, np.ndarray):
        result = array_or_scalar.copy()
        mask = ~np.isin(array_or_scalar, keep_values)
        result[mask] = np.ceil(array_or_scalar[mask]).astype(int) + 74 #- np.ceil(array_or_scalar[mask]).astype(int)
        return result
    else:
        raise TypeError("Input must be a scalar (int, float) or a NumPy array.")


# MY NEW WAY
def compute_local_maxima(volume, distance_map, max_filtered):
    
    distance_map = distance_map.copy()
    # Step 1: Check dimensions
    if volume.ndim not in [2, 3]:
        raise Exception("Make sure np.array has 2 or 3 dimensions.")

    # Step 2: Initialize local maxima mask
    local_max = np.zeros_like(distance_map, dtype=bool)
    visited = np.zeros_like(distance_map, dtype=bool)  

    # Step 3: Get all nonzero coordinates sorted by highest distance transform values
    coords = np.column_stack(np.nonzero(distance_map))  # Get all nonzero coordinates
    sorted_coords = coords[np.argsort(distance_map[tuple(coords.T)])[::-1]]  # Sort by descending distance

    # Step 4: Process Each Candidate
    for coord in sorted_coords:
        if visited[tuple(coord)]:  
            continue  # Skip already visited points

        # Get distance transform value (radius for neighborhood check)
        radius = int(distance_map[tuple(coord)])
        if radius == 0:
            continue  # Skip points with zero distance

        # Define neighborhood bounds (square)
        slices = tuple(
            slice(max(c - radius, 0), min(c + radius + 1, volume.shape[i]))
            for i, c in enumerate(coord)
        )

        # Extract local region and visited mask
        local_patch = distance_map[slices]
        local_visited = visited[slices]  # Extract visited mask in the region

        # Ensure we do not count already visited areas
        local_patch[local_visited] = 0  # Exclude previously visited areas

        # Check if current point is the maximum within its square neighborhood
        if distance_map[tuple(coord)] >= np.max(local_patch) and distance_map[tuple(coord)] == max_filtered[tuple(coord)]:
            local_max[tuple(coord)] = True  # Mark as a local maximum
            visited[slices] = True  # Mark the whole neighborhood as visited

    return local_max

"""
# FRAMEWORK WAY
from skimage.feature import peak_local_max
def compute_local_maxima(volume, distance_map, max_filtered, min_distance=3):

    # Check dimensions
    if volume.ndim not in [2, 3]:
        raise ValueError("Input volume must have 2 or 3 dimensions.")
    # Find local peaks with a minimum separation distance
    local_max_coords = peak_local_max(
        distance_map,
        min_distance=min_distance,  # Ensures well-separated centers
        indices=True,               # Returns coordinates
        exclude_border=False        # Includes borders if needed
    )
    # Create a boolean mask with peaks set to True
    local_max = np.zeros_like(distance_map, dtype=bool)
    local_max[tuple(local_max_coords.T)] = True

    return local_max
"""
"""
# MY BASIC WAY
def compute_local_maxima(volume, distance_map, max_filtered):

    
    # Check dimensions
    if volume.ndim not in [2, 3]:
        raise Exception("Make sure np.array has 2 or 3 dimensions.")

    # Compute gradient only where dimension > 1
    gradient_x = np.gradient(distance_map, axis=0) if distance_map.shape[0] > 1 else np.zeros_like(distance_map)
    gradient_y = np.gradient(distance_map, axis=1) if distance_map.shape[1] > 1 else np.zeros_like(distance_map)
    
    if volume.ndim == 3:
        gradient_z = np.gradient(distance_map, axis=2) if distance_map.shape[2] > 1 else np.zeros_like(distance_map)
        local_max = (
            ((gradient_x == 0) & (gradient_y == 0)) |
            ((gradient_y == 0) & (gradient_z == 0)) |
            ((gradient_x == 0) & (gradient_z == 0))
        ) & (distance_map == max_filtered) & (distance_map > 0)

    elif volume.ndim == 2:
        local_max = (
            (gradient_x == 0) & (gradient_y == 0)
        ) & (distance_map == max_filtered) & (distance_map > 0)

    return local_max
"""

def Watershed(volume):
    print("Distance transform")
    # Step 2: Compute the distance transform
    distance_map = ndi.distance_transform_edt(volume)
    # Apply a maximum filter to find peak regions
    print("max filter")
    max_filtered = maximum_filter(distance_map, size=3)
    # Find local maximas
    print("compute local maxima")
    local_max = compute_local_maxima(volume, distance_map, max_filtered)    
    # Step 4: Label markers for local maximuns
    markers, max_label = ndi.label(local_max)
    # Step 5: Apply Watershed Segmentation
    distance_map = -distance_map
    #distance_map = np.log(distance_map+1)
    print("watersheds")
    connected_labels =  watershed(distance_map, markers, mask=volume)
    connected_labels = value_2_LBPM_class(connected_labels, keep_values=())
    
    # Move labels as LBPM classes:
    connected_labels = connected_labels
    # Step 6: Extract Sub-Arrays for Each Label
    valid_labels = np.unique(connected_labels)[1:] # Remove background label
        
    return connected_labels, valid_labels, distance_map, local_max

def Set_Solid_to_Solid_Label(hollow_volume, connected_labels, valid_labels, solid_cell=0, fluid_default=1):
    # Where the cell in hollow_volume is a solid or sample
    non_fluid_mask = (hollow_volume != fluid_default)
    
    # Substitute by the class
    array = np.where(non_fluid_mask, connected_labels, fluid_default).astype(np.uint8)
    
    return array

"""
# Ensure that the labels are set to the surface solid cells
def Set_Solid_to_Void_Label(hollow_volume, connected_labels,valid_labels, solid_cell=0, fluid_default=1):
    
    result = hollow_volume.copy() # Avoid comparing already set labels as fluid or samples 
    
    for i in range(hollow_volume.shape[0]):  
        for j in range(hollow_volume.shape[1]):  
            for k in range(hollow_volume.shape[2]): 
                
                element = hollow_volume[i, j, k]
                
                # If its a non-fluid cell: analyse to assign to label
                if element != fluid_default:
                    
                    neighbors = Get_Neighbors(connected_labels, i, j, k)
      
                    for item in neighbors:      
                        if item in valid_labels:
                            # If any neighbor is a non-solid (label): copy the value
                            result[i, j, k] = item
    return result
"""
def Set_Solid_to_Void_Label(hollow_volume, connected_labels, valid_labels, connectivity=1, solid_cell=0, fluid_default=1):
    """
    Efficiently assigns labels to solid cells based on neighboring valid labels.
    """
    result = hollow_volume.copy()  # Avoid modifying original array

    # Create a mask for non-fluid cells (cells that need processing)
    mask = (hollow_volume != fluid_default)

    # Get all indices of non-fluid cells
    indices = np.argwhere(mask)

    # Process only the required cells
    for i, j, k in indices:
        # Get surroding labels of the solid cell
        neighbors = Get_Neighbors(connected_labels, i, j, k)

        # Find a valid label in neighbors
        valid_neighbor_labels = [item for item in neighbors if item in valid_labels]
        
        if valid_neighbor_labels:
            # Assign the first valid label found
            result[i, j, k] = valid_neighbor_labels[0]  

    return result
    
"""
def Get_Neighbors(array, i, j, k, with_coords=False, flag_out_bounds=False):
    dim = array.shape
    i_max, j_max, k_max = dim[0] - 1, dim[1] - 1, dim[2] - 1
    neighbors = []

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                
                if di == 0 and dj == 0 and dk == 0:
                    continue  # Skip the current cell

                ni, nj, nk = i + di, j + dj, k + dk

                if 0 <= ni <= i_max and 0 <= nj <= j_max and 0 <= nk <= k_max:
                    if with_coords:
                        neighbors.append((ni, nj, nk, array[ni][nj][nk]))
                    else:
                        neighbors.append(array[ni][nj][nk])
                        
                elif flag_out_bounds:
                    if with_coords:
                        neighbors.append((ni, nj, nk, None))
                    else:
                        neighbors.append(None)
                
    return neighbors
"""

"""
def Get_Neighbors(array, i, j, k, with_coords=False, flag_out_bounds=False):
    dim = array.shape
    i_max, j_max, k_max = dim[0] - 1, dim[1] - 1, dim[2] - 1
    neighbors = []

    # Define main direction offsets
    directions = [
        (1, 0, 0), 
        (-1, 0, 0),
        (0, 1, 0), 
        (0, -1, 0),
        (0, 0, 1), 
        (0, 0, -1),
    ]

    for di, dj, dk in directions:
        ni, nj, nk = i + di, j + dj, k + dk

        if 0 <= ni <= i_max and 0 <= nj <= j_max and 0 <= nk <= k_max:
            if with_coords:
                neighbors.append((ni, nj, nk, array[ni][nj][nk]))
            else:
                neighbors.append(array[ni][nj][nk])
        elif flag_out_bounds:
            if with_coords:
                neighbors.append((ni, nj, nk, None))
            else:
                neighbors.append(None)

    return neighbors
"""
def Get_Neighbors(array, i, j, k, connectivity=1, with_coords=False, flag_out_bounds=False):

    assert array.ndim == 3, "Only 3D arrays supported"
    assert connectivity in [1, 2, 3], "Connectivity must be 1, 2, or 3"

    # Predefined direction sets
    CONNECTIVITY_DIRECTIONS = {
        1: [  # 6-connected (face neighbors)
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ],
        2: [  # 18-connected (face + edge)
            (di, dj, dk)
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            for dk in [-1, 0, 1]
            if (di, dj, dk) != (0, 0, 0) and 1 <= abs(di) + abs(dj) + abs(dk) <= 2
        ],
        3: [  # 26-connected (full cube)
            (di, dj, dk)
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            for dk in [-1, 0, 1]
            if (di, dj, dk) != (0, 0, 0)
        ]
    }

    directions = CONNECTIVITY_DIRECTIONS[connectivity]
    shape = array.shape
    neighbors = []

    for di, dj, dk in directions:
        ni, nj, nk = i + di, j + dj, k + dk

        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            value = array[ni, nj, nk]
        elif flag_out_bounds:
            value = None
        else:
            continue

        if with_coords:
            neighbors.append((ni, nj, nk, value))
        else:
            neighbors.append(value)

    return neighbors

def Remove_Internal_Solid(array, fluid_default_value=1, keep_boundary=False, connectivity=3):
    """
    Remove internal solid cells that are not adjacent to fluid or (optionally) domain boundary.
    This version uses the custom Get_Neighbors function.
    """
    new_array = array.copy()

    solid_indices = np.argwhere(array != fluid_default_value)

    for i, j, k in solid_indices:
        neighbors = Get_Neighbors(array, i, j, k, flag_out_bounds=True, connectivity=connectivity)

        has_fluid_neighbor = False
        has_boundary_neighbor = False

        for val in neighbors:
            if val == fluid_default_value:
                has_fluid_neighbor = True
                break
            if keep_boundary and val is None:
                has_boundary_neighbor = True

        if not has_fluid_neighbor and not has_boundary_neighbor:
            new_array[i, j, k] = fluid_default_value  # mark internal solid as fluid

    return new_array

"""
def Remove_Internal_Solid(array, fluid_default_value=1, keep_boundary=False, connectivity=3):
    # Create array to work on
    new_array = array.copy()
    
    # Iterate over dimensions
    dim = array.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                # If the current cell is not fluid (samples and solid):
                if array[i][j][k] != fluid_default_value:
                    # Get surrounding values
                    neighbors = Get_Neighbors(array, i, j, k, flag_out_bounds=True, connectivity=connectivity)
                    
                    any_fluid_neighbor = any((n == fluid_default_value) for n in neighbors)
                    any_boundary_neighbor = any((n == None) for n in neighbors)
                    
                    # If any neighbor is fluid or boundary (if it must be kept too), the cell is not internal solid and must be kept as it is
                    if any_fluid_neighbor or (keep_boundary and any_boundary_neighbor):
                        continue  # Keep the current not internal (surface) solid cell
                    # If not, it is an internal solid and must be set to fluid
                    else:
                        new_array[i][j][k] = fluid_default_value  # Set internal solid to fluid (fluid are not interpolated)

    return new_array
"""
def euclidean(array1, array2):
    return np.linalg.norm(array1 - array2)
      
def NN_INTERPOLATION(sub_volume, samples_coord, distance_function, solid_default_value=0):
    
    hollow_sub_volume = Remove_Internal_Solid(sub_volume)
    
    # Get coordinates [(x,y,z),...] for all solid samples
    domain_solid_coord = np.argwhere(hollow_sub_volume==solid_default_value)
    
    # Create a table with distances where columns refer to solid cells and rows refer to samples
    table = []
    for sample_x,sample_y, sample_z in samples_coord:
        
        table_row = []
        print(f"Analyzing sample ({sample_x},{sample_y}, {sample_z})")
        for i,(domain_x, domain_y, domain_z) in enumerate(domain_solid_coord):
            
            dist = distance_function( np.array([sample_x, sample_y, sample_z]),  np.array([domain_x, domain_y, domain_z]) )
            
            if (i / len(domain_solid_coord)) * 100  % 5 == 0:
                print(f"Analyzing sample ({sample_x},{sample_y}, {sample_z}): ", (i / len(domain_solid_coord)) * 100)
            
            
            table_row.append(dist)
        table.append(table_row)
    
    table = np.array(table)
    
    min_sample_distance_indexes = np.argmin(table, axis=0)
    
    result = sub_volume.copy()
    for min_dist_sample_index, (d_x, d_y, d_z)  in zip(min_sample_distance_indexes, domain_solid_coord):
        min_sample_x, min_sample_y, min_sample_z = samples_coord[min_dist_sample_index]
        result[d_x, d_y, d_z] = sub_volume[min_sample_x, min_sample_y, min_sample_z]
        
    return result 
 
def WATERSHED_GRAIN_INTERPOLATION(volume, samples_coord, distance_function):
    solid_default_value=0
    fluid_default=1
    
    # CLUSTER GRAINS
    solid_volume = (volume != fluid_default).astype(int)
    connected_labels, valid_labels, distance_map, local_max = Watershed(solid_volume)
    
    local_max_coord = np.argwhere(local_max)

    # Create a table with distances where columns refer to solid cells and rows refer to samples
    table = []
    for sample_x,sample_y, sample_z in samples_coord:
        table_row = []
        for i,(local_x, local_y, local_z) in enumerate(local_max_coord):
            dist = distance_function( np.array([sample_x, sample_y, sample_z]),  np.array([local_x, local_y, local_z]) )
            table_row.append(dist)
        table.append(table_row)
    table = np.array(table)
    
    min_sample_distance_indexes = np.argmin(table, axis=0)

    result = volume.copy()
    for min_dist_sample_index, (l_x, l_y, l_z)  in zip(min_sample_distance_indexes, local_max_coord):
        min_sample_x, min_sample_y, min_sample_z = samples_coord[min_dist_sample_index] # Get the samples coordinate
        result[l_x, l_y, l_z] = volume[min_sample_x, min_sample_y, min_sample_z] # Copy the angle value from volume 
            
    # Get all True coordinates and labels from solid `volume`
    hollow_sub_volume = Remove_Internal_Solid(volume)
    solid_coords = np.argwhere(hollow_sub_volume==solid_default_value)  # Shape (N, 3)
    solid_coords_labels = connected_labels[solid_coords[:, 0], solid_coords[:, 1], solid_coords[:, 2]]
    
    # Extract labels for max locals
    local_max_labels = connected_labels[local_max_coord[:, 0], local_max_coord[:, 1], local_max_coord[:, 2]]
    
    # Create a mapping from label to corresponding local maxima coordinate
    label_to_local_max = {label: tuple(coord) for label, coord in zip(local_max_labels, local_max_coord)}
    # Assign values based on the nearest local maxima
    for (d_x, d_y, d_z), cell_label in zip(solid_coords, solid_coords_labels):
        if cell_label in label_to_local_max:
            l_x, l_y, l_z = label_to_local_max[cell_label]
            result[d_x, d_y, d_z] = result[l_x, l_y, l_z]  # Copy value from local maxima

    return result 



"""
def KRIGING_INTERPOLATION(sub_volume, samples_coord, distance_function, solid_default_value=0):
    hollow_sub_volume = Remove_Internal_Solid(sub_volume)
    
    # Get coordinates [(x,y,z),...] for all solid samples
    domain_solid_coord = np.argwhere(hollow_sub_volume==solid_default_value)
    
    # Create a table with distances where columns refer to solid cells and rows refer to samples
    dist_table = []
    for sample_x,sample_y, sample_z in samples_coord:
        dist_table_row = []
        print(f"Analyzing sample ({sample_x},{sample_y}, {sample_z})")
        for i,(domain_x, domain_y, domain_z) in enumerate(domain_solid_coord):
            
            dist = distance_function( np.array([sample_x, sample_y, sample_z]),  np.array([domain_x, domain_y, domain_z]) )
            dist_table_row.append(dist)
            if (i / len(domain_solid_coord)) * 100  % 5 == 0:
                print(f"Analyzing sample ({sample_x},{sample_y}, {sample_z}): ", (i / len(domain_solid_coord)) * 100)
            
        dist_table.append(dist_table_row)
    dist_table = np.array(dist_table)
    
    
    samples_dist_table = []
    for sample1_x,sample1_y, sample1_z in samples_coord:
        samples_dist_table_row = []
        for sample2_x,sample2_y, sample2_z in samples_coord:
            dist = distance_function( np.array([sample1_x, sample1_y, sample1_z]),  np.array([sample2_x, sample2_y, sample2_z]) )
            samples_dist_table_row.append(dist)
        samples_dist_table.append(samples_dist_table_row)
    samples_dist_table = np.array(samples_dist_table)
"""
"""
import numpy as np
from skgstat import Variogram
from scipy.spatial.distance import cdist
from scipy.linalg import solve

def KRIGING_INTERPOLATION(sub_volume, samples_coord, sample_values, distance_function, solid_default_value=0):

    hollow_sub_volume = Remove_Internal_Solid(sub_volume)
    
    # Get coordinates [(x,y,z),...] for all solid samples
    domain_solid_coord = np.argwhere(hollow_sub_volume == solid_default_value)
    
    # Create a distance matrix between samples and solid domain points
    dist_table = cdist(samples_coord, domain_solid_coord, metric=distance_function)

    # Create a distance matrix between samples (needed for variogram modeling)
    samples_dist_table = cdist(samples_coord, samples_coord, metric=distance_function)
    
    # Fit a variogram model using sample coordinates and values
    V = Variogram(samples_coord, sample_values, model='spherical')
    
    # Construct Kriging system matrix
    n = len(samples_coord)
    K = np.ones((n + 1, n + 1))  # Kriging matrix (with Lagrange multiplier)
    K[:-1, :-1] = V.transform(samples_dist_table)  # Variogram distances
    K[-1, -1] = 0  # Last row/column for Lagrange multiplier
    
    # Prepare interpolation results
    interpolated_values = np.zeros(len(domain_solid_coord))
    
    for j in range(len(domain_solid_coord)):
        # Right-hand side vector (distances to known samples)
        rhs = np.ones(n + 1)
        rhs[:-1] = V.transform(dist_table[:, j])  # Use precomputed dist_table
        
        # Solve for Kriging weights
        weights = solve(K, rhs)
        
        # Interpolate value
        interpolated_values[j] = np.sum(weights[:-1] * sample_values)
    
    # Reshape into the original volume shape
    interpolated_volume = np.full(sub_volume.shape, np.nan)
    for (x, y, z), value in zip(domain_solid_coord, interpolated_values):
        interpolated_volume[x, y, z] = value
    
    return interpolated_volume
"""
  
    
    
    

def SAMPLE_EXPANSION_INTERPOLATION(sub_volume, samples_coord, solid_default_value=0):
    
    # Limit expansion to surface by removing internal solid
    hollow_sub_volume = Remove_Internal_Solid(sub_volume, connectivity=3) # Keep a thick surface, keeping solids with any direction's fluid neighbors 

    result_volume = hollow_sub_volume.copy()
    
    # Initialize queue for BFS expansion
    to_expand_coords = samples_coord.copy()
    

    # Progress tracking setup
    """
    save_interval = 5  # percent
    total_solid_cells = max(np.count_nonzero(hollow_sub_volume == solid_default_value), 1)  # avoid division by 0
    i_counter = 0
    range_index = 0
    ranges = np.arange(0, 100+2*save_interval, save_interval)
    visited_ranges = []
    frame_names = [] 
    """
    
    while len(to_expand_coords) > 0:
                
        # Get first element and remove it from queue
        i, j, k = to_expand_coords[0]
        to_expand_coords = to_expand_coords[1:]  # Remove first element        
        if result_volume[i, j, k] == solid_default_value: quit


        # Extract neighbor coordinates and values
        neighborhood = Get_Neighbors(result_volume, i, j, k, with_coords=True, connectivity=1) # Move only in main direction 
        if len(neighborhood) == 0: continue  # Skip if there are no neighbors
        neighborhood = np.array(neighborhood)
        n_coords = neighborhood[:, :3].astype(int)  # (n_i, n_j, n_k)
        n_values = neighborhood[:, 3]  # Neighbor values

        # Not visited neighbor cells (solid cells)
        n_coords = n_coords[(n_values == solid_default_value)]
        n_values = n_values[(n_values == solid_default_value)]
        
        if len(n_coords)>0:  # If there are valid neighbors
            # Assign all neighbors to the expanded cell value
            result_volume[tuple(np.array(n_coords.T))] = result_volume[i, j, k]
            to_expand_coords.extend(n_coords)
    
    
    """
        # Progress tracking
        progress = (100 * i_counter) / total_solid_cells
        i_counter +=1
        if ranges[range_index] <= progress < ranges[range_index+1] and not (range_index in visited_ranges):
            visited_ranges.append(range_index)
            range_index +=1
            if volume_shape[0] == 1:
                image_file = pl.plot_classified_map(result_volume, f"aux_{i_counter}")
            else:
                image_file = pl.Plot_Classified_Domain(result_volume, f"aux_{i_counter}", remove_value=[1])
            frame_names.append(image_file)
    pl.create_gif_from_filenames(
        image_filenames = frame_names, 
        gif_filename="TEST", 
        duration=round(10000/len(frame_names)), 
        loop=0, 
        erase_plots=True)
    """
    
    return result_volume
    
    

def COLLECT_SAMPLES_COORDINATES(sub_array):
    samples_coord = []
    for i in range(sub_array.shape[0]):  
        for j in range(sub_array.shape[1]):  
            for k in range(sub_array.shape[2]):  
                
                # acess element
                element = sub_array[i, j, k]

                # Collect samples infos (Non-solid and non-fluid):
                if is_LBPM_class(element):
                    samples_coord.append([i, j, k])
                    
    return samples_coord
                    
def Keep_random_samples(volume, solid_value=0, fluid_value=1, N=0.1):

    hollow_volume = Remove_Internal_Solid(volume)
    
    # Where samples are (surface)
    samples_mask = (hollow_volume != solid_value) & (hollow_volume != fluid_value)
    
    # Index of where samples are
    samples_mask_index = np.argwhere(samples_mask)
    
    total_amount_samples = len(samples_mask_index)
    N_kept_samples = int(N*total_amount_samples)

    # Select N indices to keep (if N is greater than available, keep all)
    keep_samples_mask_index = samples_mask_index[np.random.choice(len(samples_mask_index), N_kept_samples, replace=False)]
    
    # Create a copy of the volume
    modified_volume = volume.copy()

    # Fill all sample positions with `fill_value`
    non_fluid_mask = (volume != fluid_value)
    modified_volume[non_fluid_mask] = solid_value

    # Restore the N selected positions to their original values
    modified_volume[tuple(keep_samples_mask_index.T)] = volume[tuple(keep_samples_mask_index.T)]
    
    return modified_volume

def Change_Interpolated_Cells(final_volume, interpolated_volume, fluid_default_value=1):

    changes_mask = interpolated_volume != fluid_default_value
    final_volume[changes_mask] = interpolated_volume[changes_mask]
    
    """
    (DEPRECATED C-LIKE CODE)
    #for i in range(interpolated_volume.shape[0]):
        #for j in range(interpolated_volume.shape[1]):
            #for k in range(interpolated_volume.shape[2]):
                # Where there was interpolation
                #if interpolated_volume[i,j,k] != fluid_default_value:
                    #final_volume[i,j,k] = interpolated_volume[i,j,k]
    """
    return final_volume
                    

        

def GET_INTERPOLATED_DOMAIN(sampled_volume, interpolation_mode, fluid_default_value=1, solid_default_value=0):

    # FOR EACH SUB DOMAIN
    final_volume = sampled_volume.copy()
    
    # COLLECT SAMPLES INFOS 
    samples_coord = COLLECT_SAMPLES_COORDINATES(sampled_volume)
    
    # If there is no sample in the domain, neglect it
    if samples_coord:
        
        print("Calculating distance transform")
        # GET THE DISTANCE OF EVERY CELL TO THE SUB-VOLUME SAMPLES
        #if distance_mode=="euclidean":
        #    distance_function = euclidean
        #elif distance_mode=="path":
        #    obj = surface_distance(sub_volume, samples_coord) 
        #    distance_function = obj.get_distance     
        #else:
        #    raise Exception("Not Implemented.")
                     
        distance_function = euclidean
        
        print("Interpolating")
        #  INTERPOLATE
        if interpolation_mode == "nn":
            interpolated_volume = NN_INTERPOLATION(sampled_volume, samples_coord, distance_function, solid_default_value)
        elif interpolation_mode == "watershed_grain":
            interpolated_volume = WATERSHED_GRAIN_INTERPOLATION(sampled_volume, samples_coord, distance_function)
        elif interpolation_mode == "expand_samples":
            interpolated_volume = SAMPLE_EXPANSION_INTERPOLATION(sampled_volume, samples_coord)
        #elif interpolation_mode == "kriging":
        #    interpolated_volume = KRIGING_INTERPOLATION(sub_volume, samples_coord, solid_default_value)
        else:
            raise Exception("Not Implemented.")
        print("Interpolation finished")
        
        
        # Assign interpolated values to the right places
        final_volume = Change_Interpolated_Cells(final_volume, interpolated_volume)

    return final_volume
    
def Get_Metrics(reference_volume, interpolated_volume, sampled_volume):
    if reference_volume.shape != interpolated_volume.shape:
        raise Exception(f"Shapes must match, but got {reference_volume.shape} != {interpolated_volume.shape}")
        
    # Remove internal solid areas before processing: do not count internal solid classifications
    reference_volume = Remove_Internal_Solid(reference_volume)
    interpolated_volume = Remove_Internal_Solid(interpolated_volume)
    
    # Get mask of LBPM classes (not solid or fluids)
    samples_mask = is_LBPM_class(sampled_volume) # Samples should not count into metrics
    gt_classified_mask = is_LBPM_class(reference_volume) & ~ samples_mask  # Ground truth classified cells
    result_classified_mask = is_LBPM_class(interpolated_volume) & ~ samples_mask # Interpolated classified cells
    valid_mask = gt_classified_mask | result_classified_mask  # Union of classified cells
    
    # Get mask of cells correctly classified
    right_classified_mask = gt_classified_mask & (reference_volume == interpolated_volume)
    
    # **Compute Metrics Using Vectorized Operations**
    count_gt_classified = np.count_nonzero(gt_classified_mask)
    count_classifieds = np.count_nonzero(valid_mask)
    count_success = np.count_nonzero(right_classified_mask)

    # Compute Absolute and Percentage Errors
    abs_errors = np.abs(reference_volume - interpolated_volume) * valid_mask
    summed_abs_e = np.sum(abs_errors)
    
    # **Compute Final Metrics**
    acc = count_success / count_gt_classified if count_gt_classified > 0 else 0
    iou = count_success / count_classifieds if count_classifieds > 0 else 0
    mae = summed_abs_e / count_classifieds if count_classifieds > 0 else 0
    
    return {
        "Accuracy": acc, 
        "IOU": iou,
        "MAE": mae,
    }




"""
(DEPRECATED USAGE)

class surface_distance():
    
    def __init__(self, volume, samples_coord=None):
        self.field_binary = ~volume.astype(bool) # boolean field for seach (0->True is the search area) 
        self.samples_coord = samples_coord
        self.pre_processed = samples_coord is not None
        self.parents = []
        
        
        if self.pre_processed:
            for sample_x,sample_y,sample_z in samples_coord:              
                self.field_binary = self.field_binary.astype(int)
                source_field = dijkstra3d.parental_field(volume, source=(sample_x,sample_y,sample_z), connectivity=6)
                
                self.parents.append(
                    {"source_field": source_field,
                     "source_coordinate": (sample_x,sample_y,sample_z)
                        })
                
            
    def get_distance(self, source, target):
        
        path = np.array([])

        # Find path
        if self.pre_processed: 
            for source_infos in self.parents:
                sample_x, sample_y, sample_z = source_infos["source_coordinate"]
                source_field = source_infos["source_field"]
                if sample_x==source[0] and sample_y==source[1] and sample_z==source[2]:
                    path = dijkstra3d.path_from_parents(source_field, target=(target[0], target[1], target[2]))
        else:
            path = dijkstra3d.binary_dijkstra(self.field_binary, source, target, connectivity=26, background_color=0,euclidean_metric=True)



        # If there is no available path set the distance to infinit
        if path.size > 0:
            distance = self.calculate_path_size(path)
            return distance 
        else:
            return np.inf
    
    def calculate_path_size(self, path, approximate=False):    
        if approximate==True: return len(path)
        return sum( euclidean(path[i], path[i-1]) for i in range(1,len(path)) )
  
def CLUSTER(volume, cluster_mode='grains',fluid_default=1, file_name=""):
    
    if cluster_mode=='grains':
        # Convert to binary: 1 for solid, 0 for void (fluid)
        solid_volume = (volume != fluid_default).astype(np.uint8)
        
        connected_labels, valid_labels, distance_map, local_max = Watershed(solid_volume, file_name)
        
        volume_labelled = Set_Solid_to_Solid_Label(volume, connected_labels, valid_labels)

        
    elif cluster_mode=='pores':
        # Convert to binary: 0 for solid, 1 for void (fluid)
        solid_volume = 1-(volume != fluid_default).astype(np.uint8)
        
        connected_labels, valid_labels, distance_map, local_max = Watershed(solid_volume, file_name)
        
        # Set the interfaced solid to the void label
        volume_labelled = Set_Solid_to_Void_Label(volume, connected_labels, valid_labels)
        
    elif cluster_mode == 'none':
        return  [volume], [0] 
        
    else:
        raise Exception("Unkown clustering mode {mode}. Choose grains or pores as mode.")
        
    
    # Save infos
    sub_arrays = []
    for label_value in valid_labels:
        # Mask only the desired cluster
        mask = (volume_labelled == label_value)
        # Get sub-array with just the solid cells and samples of the cluster
        sub_array = np.where(mask, volume, fluid_default).astype(np.uint8)
        # Do not save clusters with only fluid cells
        if np.any(sub_array!= fluid_default):
            sub_arrays.append(sub_array)
            
    
    if file_name != "":        
        Plot_Classified_Domain(connected_labels, file_name+"_conn_sol", remove_value=[1,0])
        Plot_Classified_Domain(hollow_volume_labelled, file_name+"_hollow_conn_sol", remove_value=[1,0])
        plot_classified_map(hollow_volume[0,:,:], file_name+"_hollow_volume_2D")
        plot_classified_map(connected_labels[0,:,:], file_name+"_conn_sol_2D")
        plot_classified_map(hollow_volume_labelled[0,:,:], file_name+"_hollow_conn_sol_2D")
    
    return sub_arrays, valid_labels
"""

    

         