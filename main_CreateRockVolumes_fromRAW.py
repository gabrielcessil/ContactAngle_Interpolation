import numpy as np
import random
import Plotter as pl
import utilities as util

def extract_non_overlapping_slices(volume, slice_shape, N=None):

    size_x, size_y, size_z = volume.shape
    dx, dy, dz = slice_shape  # Desired slice size
    
    slices = []
    
    # Iterate through the volume with step = slice size (no overlap)
    
    count = 0
    for x in range(0, size_x, dx):
        for y in range(0, size_y, dy):
            for z in range(0, size_z, dz):
                # Ensure slice fits within bounds
                if x + dx <= size_x and y + dy <= size_y and z + dz <= size_z:
                    
                    if not N is None and count > N:
                        return slices
                    
                    slices.append(volume[x:x+dx, y:y+dy, z:z+dz])
                    
                    count+=1

    return slices

def filter_connected_labels(connected_labels, valid_labels, sub_shape=()):

    volume = connected_labels.copy()
    # 1. Create a label count dictionary for the entire array
    label_counts = {}
    for value in np.unique(volume):
        label_counts[value] = np.count_nonzero(volume == value)

    # 2. Iterate through all non-overlapping sub-volumes
    x_max, y_max, z_max,  = volume.shape
    for x_start in range(0, x_max, max(sub_shape[0], 1)):
        for y_start in range(0, y_max, max(sub_shape[1], 1)):
            for z_start in range(0, z_max, max(sub_shape[2], 1)):
                
                x_end = min(x_max, x_start + sub_shape[0])
                y_end = min(y_max, y_start + sub_shape[1])
                z_end = min(z_max, z_start + sub_shape[2])
    
                sub_volume = volume[x_start:x_end, y_start:y_end, z_start:z_end].copy()
                #pl.plot_classified_map(sub_volume, file_path+f"_subvolume_{z_end}_{y_end}_{x_end}",  colormap='tab20')
                
                # 3. Analyze sub-volume valid labels and counts
                labels_inSubVolume = np.intersect1d(valid_labels, np.unique(sub_volume))

                # 4. Find most common label
                count_labels_inSubVolume = {}
                for label in labels_inSubVolume:
                    count_labels_inSubVolume[label] = np.count_nonzero(sub_volume == label)
    
                if count_labels_inSubVolume: #check if the subvolume is not empty.
                    most_common_label = max(count_labels_inSubVolume, key=count_labels_inSubVolume.get)
                    
                    # 5. Replace labels based on sub-volume dominance (only valid labels can be replaced, not background)
                    for label in  np.intersect1d(labels_inSubVolume, valid_labels):
                        # If all the label's cluster is inside the sub-volume
                        total_count = np.count_nonzero(volume == label)
                        if count_labels_inSubVolume[label] == total_count:
                            sub_volume[sub_volume == label] = most_common_label
                            #pl.plot_classified_map(sub_volume, file_path+f"_subvolume_{z_end}_{y_end}_{x_end}_Rep",  colormap='tab20')
                    
                    # 6. Set the changed labels to the volume
                    volume[x_start:x_end, y_start:y_end, z_start:z_end] = sub_volume #assign the subvolume back to the original volume.
    
    new_valid_labels = np.intersect1d(valid_labels, np.unique(volume)) 
    
    return volume, new_valid_labels
                    

# Loading entire rock
input_file_name = r"Rock Volumes/Bentheimer/bentheimer.raw"
volume_shape = (1000,1000,1000)
volume = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
# Setting Grain(solid) = 0, Fluid = 1
volume = volume/255
volume = 1-volume


N = 10 
slice_shape = (200, 200, 200)

cluster_mode = 'grains'
angles = [45, 135]
labels = {  0: "Original Rock",
            1: "Void Space",
            45: "Water-Wetting Cells (30º)",
            135: "Oil-Wetting Cells (150º)"}
special_colors = {
    0: (0.5, 0.5, 0.5, 1.0),
    1: (0.0, 0.0, 0.0, 1.0)
}
fluid_default = 1
solid_default = 0


# Converting labels values to LBPM class
labels = {util.value_2_LBPM_class(angle): label for angle, label in labels.items()}
special_colors = {util.value_2_LBPM_class(angle): color for angle, color in special_colors.items()}

slices = extract_non_overlapping_slices(volume, slice_shape, N=N)

random.seed(42)  # Use any integer you like; 42 is a common example
for i, volume_slice in enumerate(slices):
    
    print("Creating domain ", i)
    volume_final = volume_slice.copy()
    s_x, s_y, s_z = volume_slice.shape
    file_path = f"Rock Volumes/Bentheimer_Filtered/{cluster_mode}/benthheimer_{s_x}x{s_y}x{s_z}__{i}"
    
    if cluster_mode == 'grains':
        volume_to_cluster = (volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
        
    elif cluster_mode == 'pores':
        volume_to_cluster = 1-(volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
        hollow_volume = util.Remove_Internal_Solid(volume_slice, connectivity=3)
        connected_labels = util.Set_Solid_to_Void_Label(hollow_volume, connected_labels, valid_labels, connectivity=3)
        
        non_surface_solid_mask = (volume_slice != fluid_default) & (hollow_volume!=solid_default)
        fluid_mask = volume_slice == fluid_default
        connected_labels[non_surface_solid_mask] = min(valid_labels)-1 # Set the solids to a unvalid label, so it dont get in raffle
        connected_labels[fluid_mask] = min(valid_labels)-2 # Set the fluid to a unvalid label, so it dont get in raffle
    else:
        raise Exception("Not implemented")
    
        
    # Filter result
    filter_cube = np.maximum(np.array(slice_shape)/2,1)
    connected_labels, valid_labels = filter_connected_labels(connected_labels, valid_labels, filter_cube.astype(int))
    
    # Raffle contact angles across clusters
    for label in valid_labels:
        angle = random.choice(angles) # Choose one angle
        angle_class = util.value_2_LBPM_class(angle)  # Convert angle to LBPM class
        group_mask = connected_labels==label # Make mask of the desired group
        volume_final[group_mask] = angle_class # Set angle
        
    # Save plots
    if slice_shape[0]==1:
        pl.plot_classified_map(connected_labels, file_path+"_clusters",  colormap='tab20')
        pl.plot_classified_map(volume_slice, file_path+"_volume")
        pl.plot_classified_map(volume_final, file_path+"_volume_final")
        
    else:
        
        pl.Plot_Classified_Domain(
            volume_slice, 
            file_path+"_volume", 
            remove_value=[1], 
            labels=labels,
            special_colors=special_colors
        )
        pl.Plot_Classified_Domain(
            volume_final, 
            file_path+"_volume_final", 
            remove_value=[1],
            labels=labels,
            special_colors=special_colors
        )
        
    # Save volume 
    volume_final = volume_final.astype(np.uint8)
    volume_final.tofile(file_path + "_volume_final.raw")
        

