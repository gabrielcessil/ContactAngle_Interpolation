import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import pdist

import traceback

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
    # Step 2: Compute the distance transform
    distance_map = ndi.distance_transform_edt(volume)
    # Apply a maximum filter to find peak regions
    max_filtered = maximum_filter(distance_map, size=3)
    # Find local maximas
    local_max = compute_local_maxima(volume, distance_map, max_filtered)    
    # Step 4: Label markers for local maximuns
    markers, max_label = ndi.label(local_max)
    # Step 5: Apply Watershed Segmentation
    distance_map = -distance_map
    connected_labels =  watershed(distance_map, markers, mask=volume)
    valid_labels = np.unique(connected_labels)[1:]
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
        neighbors = Get_Neighbors(connected_labels, i, j, k, connectivity=connectivity)

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
from scipy.optimize import curve_fit

def KRIGING_INTERPOLATION(sub_volume, samples_coord, distance_function, N, solid_default_value=0):

    print("KRIGING: ")
    print("Removing internal solid")
    samples_values = sub_volume[tuple(np.array(samples_coord).T)]

    hollow_sub_volume = Remove_Internal_Solid(sub_volume)

    domain_solid_coord = np.argwhere(hollow_sub_volume == solid_default_value)

    resampled_volume = sub_volume.copy().astype(float)
    
    print("Fitting global variogram model")
    global_variogram_params = fit_global_variogram(samples_coord, samples_values, distance_function)

    print("Analyzing cell by cell")
    for domain_x, domain_y, domain_z in domain_solid_coord:
        
        
        domain_point = np.array([domain_x, domain_y, domain_z])

        distances = [distance_function(domain_point, np.array(sample_coord)) for sample_coord in samples_coord]

        nearest_indices = np.argsort(distances)[:N]

        nearest_samples_coord = [samples_coord[i] for i in nearest_indices]
        nearest_samples_values = [samples_values[i] for i in nearest_indices]

        # 1. Use global variogram model
        variogram_model = lambda h: linear_variogram(h, *global_variogram_params)

        # 2. Create matrices 
        A = np.zeros((N, N))
        b = np.zeros(N)

        for i in range(N):
            for j in range(N):
                A[i, j] = variogram_model(distance_function(np.array(nearest_samples_coord[i]), np.array(nearest_samples_coord[j])))
            b[i] = variogram_model(distance_function(domain_point, np.array(nearest_samples_coord[i])))

        # Solve for weights
        try:
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            A += np.eye(N) * 1e-6
            weights = np.linalg.solve(A, b)
        
        # 3. Calculate interpolated values
        interpolated_value = int(np.dot(weights, nearest_samples_values))
        print(domain_x, domain_y, domain_z, "=", interpolated_value)
        
        # 4. Assign value to cell
        resampled_volume[domain_x, domain_y, domain_z] = interpolated_value

    return resampled_volume
"""

from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import multiprocessing
from functools import partial

def KRIGING_INTERPOLATION(sub_volume, samples_coord, distance_function, N, solid_default_value=0, num_workers=None):

    print("KRIGING: ")
    print("Removing internal solid")
    samples_values = sub_volume[tuple(np.array(samples_coord).T)]

    hollow_sub_volume = Remove_Internal_Solid(sub_volume)

    domain_solid_coord = np.argwhere(hollow_sub_volume == solid_default_value)

    resampled_volume = sub_volume.copy().astype(float)
    
    print("Fitting global variogram model")
    variogram_model, variogram_params = fit_global_variogram(samples_coord, samples_values, distance_function, mode='spherical')
    
    N = 5
    # --------------- MULTI-PROCESSING STARTS HERE ---------------
    print(f"Interpolating {len(domain_solid_coord)} cells using {num_workers or 'all'} cores...")
    # BASIC ATTEMPT
    """
    for i, domain_point in enumerate(domain_solid_coord):
        print(domain_point, f"  |   {i}  / {len(domain_solid_coord)}")
        domain_x, domain_y, domain_z, interpolated_value = interpolate_point(
                                                                variogram_model, 
                                                                variogram_params,
                                                                domain_point, 
                                                                samples_coord, 
                                                                samples_values, 
                                                                distance_function, 
                                                                N=5)
        
        resampled_volume[domain_x, domain_y, domain_z] = interpolated_value
    """  
    # PARALLEL ATTEMPT
    with multiprocessing.Pool(processes=10) as pool:
        func = partial(parallel_interpolate_task, 
                   variogram_model=variogram_model, 
                   variogram_params=variogram_params, 
                   samples_coord=samples_coord, 
                   samples_values=samples_values, 
                   distance_function=distance_function, 
                   N=N)
    
        results = pool.map(func, domain_solid_coord)
        
    # Efficiently assign results to resampled_volume using NumPy indexing
    results_array = np.array(results)
    coords = results_array[:, 0:3].astype(int)  # Ensure coordinates are integers
    values = results_array[:, 3]

    resampled_volume[tuple(coords.T)] = values
    return resampled_volume

def parallel_interpolate_task(domain_point, variogram_model, variogram_params, samples_coord, samples_values, distance_function, N):
    return interpolate_point(variogram_model, variogram_params, domain_point, samples_coord, samples_values, distance_function, N)


from scipy.sparse.linalg import cg
def interpolate_point(variogram_model, variogram_params, domain_point, samples_coord, samples_values, distance_function, N):
    
    try:
        domain_x, domain_y, domain_z = domain_point
        domain_point = np.array([domain_x, domain_y, domain_z])
    
        # Find the N nearest neighbors
        # Compute distances to all sample points
        #earest_indices = get_nearest_samples_indexes(domain_point, samples_coord, distance_function, N) 
        nearest_indices = np.argsort([ distance_function(domain_point, point) for point in samples_coord] )[:N]
        #print(domain_x, domain_y, domain_z)
        
        
        nearest_samples_coord = [samples_coord[i] for i in nearest_indices]
        nearest_samples_values = [samples_values[i] for i in nearest_indices]
        
        #print(nearest_samples_coord)
        #print(nearest_samples_values)
            
        # Create matrices
        
        N_total = len(nearest_indices) # How many samples were indeed considered (it may vary with availability within quadrants)
        A = np.zeros((N_total, N_total))
        b = np.zeros(N_total)
        
        estimate_semivariance = lambda dist: variogram_model(dist, *variogram_params)
        
        for i in range(N_total):
            for j in range(N_total):
                A[i, j] = estimate_semivariance(distance_function(np.array(nearest_samples_coord[i]), np.array(nearest_samples_coord[j])))
            b[i] = estimate_semivariance(distance_function(domain_point, np.array(nearest_samples_coord[i])))
    
        # Solve for weights
        #print("A: ",A)
        #print("b: ",b)
        #print(np.linalg.cond(A))
        try:
            weights = np.linalg.solve(A, b)
            #weights, conv_flag = cg(A, b)
            
        except np.linalg.LinAlgError:
            
            weights, conv_flag = cg(A, b)
            if conv_flag == 0:
                # Adds a small identity matrix to make it non-singular
                A += np.eye(N_total) * 1e-6
                weights, conv_flag = cg(A, b)
                
                if conv_flag == 0:
                    return "Error"
                
                
        #print("weights: ", weights)
        # Compute interpolated value compuitng the values * weights
        interpolated_value = int(np.dot(weights, nearest_samples_values))
        
        print("::::", domain_x, domain_y, domain_z, interpolated_value)

        return domain_x, domain_y, domain_z, interpolated_value
    
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()  # Prints the full error traceback
        return f"Error: {e}"

import matplotlib.pyplot as plt
def plot_fitted_variogram(variogram_model,popt,pairwise_distances, semivariances, binned_distances, binned_semivariances, samples_coord, samples_values, distance_function, bins):
    
    continuous_distances = np.linspace(0, np.max(binned_distances), 100)
    continuous_semivariances = variogram_model(continuous_distances, *popt)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ### **Left Plot: Semivariance vs. Pairwise Distance**
    for i, bin_edge in enumerate(bins):
        if i == 0:
            axes[0].axvline(bin_edge, color="black", linestyle="--", alpha=0.8, label="Lag")  # Bin markers
        else: 
            axes[0].axvline(bin_edge, color="black", linestyle="--", alpha=0.8)
            
    indices = np.random.choice(range(len(pairwise_distances)), size=min(len(pairwise_distances), (len(bins)-1)*300), replace=False)
    
    axes[0].scatter(pairwise_distances[indices], np.log(semivariances[indices]), color="blue", alpha=0.1, label="Raw Semivariance")
    axes[0].set_xlabel("Pairwise Distance")
    axes[0].set_ylabel("Semivariance")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)  # Move legend up
    axes[0].set_xticks(bins)  # Left plot
    axes[0].minorticks_on()
    axes[0].grid(which="minor", linestyle=":", linewidth=0.5, color="grey")

    ### **Right Plot: Empirical and Fitted Variogram**
    for bin_edge in bins:
        axes[1].axvline(bin_edge, color="black", linestyle="--", alpha=0.8)  # Bin markers
    axes[1].plot(continuous_distances, continuous_semivariances, "k-", label="Fitted Variogram Model")
    axes[1].plot(binned_distances, binned_semivariances, "r*", markersize=12,label="Average Semivariance per Lag Bin")
    
    axes[1].set_xlabel("Average Pairwise Distance per Lag Bin")
    axes[1].set_ylabel("Semivariance")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)  # Move legend up
    axes[1].set_xticks(bins)  # Right plot
    axes[1].minorticks_on()
    axes[1].grid(which="minor", linestyle=":", linewidth=0.5, color="grey")
    
    fig.suptitle("EMPIRICAL VARIOGRAM",fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    
def linear_variogram(h, slope, nugget):
    """Linear variogram model."""
    return nugget + slope * h
def spherical_variogram(h, sill, range_, nugget):
    """Spherical variogram model."""
    return np.where(h <= range_, nugget + sill * (1.5 * h / range_ - 0.5 * (h / range_)**3), nugget + sill)
def exponential_variogram(h, sill, range_, nugget):
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-3 * h / range_))
def gaussian_variogram(h, sill, range_, nugget):
    """Gaussian variogram model."""
    return nugget + sill * (1 - np.exp(-3 * (h / range_)**2))

def fit_global_variogram(samples_coord, samples_values, distance_function, n_bins=30, mode='linear'):
    """Fits a global linear variogram to all samples, using pairwise distance computation."""
    
    # Convert to numpy arrays for efficient manipulation
    samples_coord = np.array(samples_coord)
    samples_values = np.array(samples_values)
    
    # Compute pairwise distances using pdist
    pairwise_distances = pdist(samples_coord, metric=distance_function)
    
    # Compute semivariances for each pair of points
    semivariances = 0.5 * (samples_values[:, None] - samples_values)**2  # Efficient broadcasting to compute all pairs
    semivariances = semivariances[np.triu_indices_from(semivariances, k=1)]  # Upper triangle without diagonal
    
    # Determine distance bins
    min_dist = np.min(pairwise_distances)
    max_dist = np.max(pairwise_distances)
    n_bins = min(n_bins, len(samples_values))
    bins = np.linspace(min_dist, max_dist, n_bins + 1)

    # Bin the distances and semivariances
    binned_distances = []
    binned_semivariances = []
    for i in range(n_bins):
        bin_indices = np.where((pairwise_distances >= bins[i]) & (pairwise_distances < bins[i+1]))[0]
        if len(bin_indices) > 0:
            binned_distances.append(np.mean(pairwise_distances[bin_indices]))
            binned_semivariances.append(np.mean(semivariances[bin_indices]))
    # Convert to numpy arrays for curve fitting
    binned_distances = np.array(binned_distances)
    binned_semivariances = np.array(binned_semivariances)

    # Fit the variogram model to binned data
    if len(binned_distances) > 1: # check if there are enough bins to fit the model
        # Select variogram model and initial parameters
        if mode == 'linear':
            variogram_model = linear_variogram
            p0 = [1.0, 0.0]  # [slope, nugget]
        elif mode == 'spherical':
            variogram_model = spherical_variogram
            p0 = [np.max(binned_semivariances), np.max(binned_distances)/2, np.min(binned_semivariances)] # [sill, range, nugget]
        elif mode == 'exponential':
            variogram_model = exponential_variogram
            p0 = [np.max(binned_semivariances), np.max(binned_distances)/2, np.min(binned_semivariances)] # [sill, range, nugget]
        elif mode == 'gaussian':
            variogram_model = gaussian_variogram
            p0 = [np.max(binned_semivariances), np.max(binned_distances)/2, np.min(binned_semivariances)] # [sill, range, nugget]
        else:
            raise ValueError("Invalid variogram mode. Choose 'linear', 'spherical', 'exponential', or 'gaussian'.")

        popt, _ = curve_fit(variogram_model, binned_distances, binned_semivariances, p0=p0)  # Initial guesses for slope and nugget
        plot_fitted_variogram(variogram_model, popt, pairwise_distances, semivariances, binned_distances, binned_semivariances, samples_coord, samples_values, distance_function, bins)
        
        return variogram_model, popt 
    else:
        return None, None # return none if not enough data to fit the model.




# Get samples from different quadrant
def get_nearest_samples_indexes(domain_point, samples_coord, distance_function, N):
    
    coordinates = np.array(samples_coord)
    diff = samples_coord - np.array(domain_point)
    signs = np.sign(diff)
    
    # Get coordinate from each quadrant
    quad_ppp_coord = coordinates[(signs[:, 0] >= 0) & (signs[:, 1] >= 0) & (signs[:, 2] >= 0)].tolist()
    quad_ppm_coord = coordinates[(signs[:, 0] >= 0) & (signs[:, 1] >= 0) & (signs[:, 2] < 0)].tolist()
    quad_pmp_coord = coordinates[(signs[:, 0] >= 0) & (signs[:, 1] < 0) & (signs[:, 2] >= 0)].tolist()
    quad_pmm_coord = coordinates[(signs[:, 0] >= 0) & (signs[:, 1] < 0) & (signs[:, 2] < 0)].tolist()
    quad_mpp_coord = coordinates[(signs[:, 0] < 0) & (signs[:, 1] >= 0) & (signs[:, 2] >= 0)].tolist()
    quad_mpm_coord = coordinates[(signs[:, 0] < 0) & (signs[:, 1] >= 0) & (signs[:, 2] < 0)].tolist()
    quad_mmp_coord = coordinates[(signs[:, 0] < 0) & (signs[:, 1] < 0) & (signs[:, 2] >= 0)].tolist()
    quad_mmm_coord = coordinates[(signs[:, 0] < 0) & (signs[:, 1] < 0) & (signs[:, 2] < 0)].tolist()
    
    # Get the distances of this set of coordinates to the defined domain point
    quad_ppp_dist = [ distance_function(domain_point, coord) for coord in quad_ppp_coord]
    quad_ppm_dist = [ distance_function(domain_point, coord) for coord in quad_ppm_coord]
    quad_pmp_dist = [ distance_function(domain_point, coord) for coord in quad_pmp_coord]
    quad_pmm_dist = [ distance_function(domain_point, coord) for coord in quad_pmm_coord]
    quad_mpp_dist = [ distance_function(domain_point, coord) for coord in quad_mpp_coord]
    quad_mpm_dist = [ distance_function(domain_point, coord) for coord in quad_mpm_coord]
    quad_mmp_dist = [ distance_function(domain_point, coord) for coord in quad_mmp_coord]
    quad_mmm_dist = [ distance_function(domain_point, coord) for coord in quad_mmm_coord]
    
    # Sort the distances and get the indexes of the (up to) N nearest
    quad_ppp_near_i = np.argsort(quad_ppp_dist)[:min(N, len(quad_ppp_dist))]
    quad_ppm_near_i = np.argsort(quad_ppm_dist)[:min(N, len(quad_ppm_dist))]
    quad_pmp_near_i = np.argsort(quad_pmp_dist)[:min(N, len(quad_pmp_dist))]
    quad_pmm_near_i = np.argsort(quad_pmm_dist)[:min(N, len(quad_pmm_dist))]
    quad_mpp_near_i = np.argsort(quad_mpp_dist)[:min(N, len(quad_mpp_dist))]
    quad_mpm_near_i = np.argsort(quad_mpm_dist)[:min(N, len(quad_mpm_dist))]
    quad_mmp_near_i = np.argsort(quad_mmp_dist)[:min(N, len(quad_mmp_dist))]
    quad_mmm_near_i = np.argsort(quad_mmm_dist)[:min(N, len(quad_mmm_dist))]

    
    # Make a list using the provided indexes
    re_list = lambda values, indexes: [values[i] for i in indexes]
    
    quad_ppp_near_coord = re_list(quad_ppp_coord, quad_ppp_near_i) # List of coordinates
    quad_ppm_near_coord = re_list(quad_ppm_coord, quad_ppm_near_i) 
    quad_pmp_near_coord = re_list(quad_pmp_coord, quad_pmp_near_i)
    quad_pmm_near_coord = re_list(quad_pmm_coord, quad_pmm_near_i)
    quad_mpp_near_coord = re_list(quad_mpp_coord, quad_mpp_near_i)
    quad_mpm_near_coord = re_list(quad_mpm_coord, quad_mpm_near_i)
    quad_mmp_near_coord = re_list(quad_mmp_coord, quad_mmp_near_i)    
    quad_mmm_near_coord = re_list(quad_mmm_coord, quad_mmm_near_i)
    
    coord_list = quad_ppp_near_coord + quad_ppm_near_coord + quad_pmp_near_coord + quad_pmm_near_coord + quad_mpp_near_coord + quad_mpm_near_coord + quad_mmp_near_coord + quad_mmm_near_coord

    near_indexes = np.where(np.isin(list(map(tuple, samples_coord)), list(map(tuple, coord_list))))[0]
    
    return near_indexes
    


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
        elif interpolation_mode == "kriging":
            interpolated_volume = KRIGING_INTERPOLATION(sampled_volume, samples_coord, distance_function, 5, solid_default_value)
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

    

         