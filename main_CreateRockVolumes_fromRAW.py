import numpy as np
import random
import Plotter as pl
import utilities as util

def extract_non_overlapping_slices(volume, slice_shape):
    """
    Extract non-overlapping 3D slices from a given 3D array.
    
    Args:
        volume (np.array): The 3D input array.
        slice_shape (tuple): The shape (dx, dy, dz) of each slice.
    
    Returns:
        list of np.array: A list containing extracted 3D slices.
    """
    size_x, size_y, size_z = volume.shape
    dx, dy, dz = slice_shape  # Desired slice size
    
    slices = []
    
    # Iterate through the volume with step = slice size (no overlap)
    for x in range(0, size_x, dx):
        for y in range(0, size_y, dy):
            for z in range(0, size_z, dz):
                # Ensure slice fits within bounds
                if x + dx <= size_x and y + dy <= size_y and z + dz <= size_z:
                    slices.append(volume[x:x+dx, y:y+dy, z:z+dz])

    return slices



input_file_name = r"Rock Volumes/Bentheimer/bentheimer.raw"
volume_shape = (1000,1000,1000)

volume = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)

# Setting Grain(solid) = 0, Fluid = 1
volume = volume/255
volume = 1-volume
fluid_default = 1
solid_default = 0

cluster_mode = 'grains'

angles = [30, 90, 150]
labels = {  0: "Original Rock",
            30: "Water-Wetting Cells (30º)",
            90: "Neutral-Wetting Cells (90º)",
            150: "Oil-Wetting Cells (150º)"}

slice_shape = (200, 200, 200)  # Define non-overlapping slice shape


# Converting labels values to LBPM class
labels = {util.value_2_LBPM_class(angle): label for angle, label in labels.items()}
print("Extracting non-overlapping slices")
slices = extract_non_overlapping_slices(volume, slice_shape)
random.seed(42)  # Use any integer you like; 42 is a common example
for i, volume_slice in enumerate(slices):
    
    print("Creating domain ", i)
    volume_final = volume_slice.copy()
    s_x, s_y, s_z = volume_slice.shape
    file_path = f"Rock Volumes/Bentheimer/{cluster_mode}/benthheimer_{s_x}x{s_y}x{s_z}__{i}"
    
    if cluster_mode == 'grains':
        volume_to_cluster = (volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
        
    elif cluster_mode == 'pores':
        volume_to_cluster = 1-(volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
        hollow_volume = util.Remove_Internal_Solid(volume_slice)
        connected_labels = util.Set_Solid_to_Void_Label(hollow_volume, connected_labels, valid_labels, connectivity=3)

    else:
        raise Exception("Not implemented")


    # Raffle contact angles across clusters
    print("Contact Angle Raffle")
    for label in valid_labels:
        angle = random.choice(angles) # Choose one angle
        angle_class = util.value_2_LBPM_class(angle)  # Convert angle to LBPM class
        group_mask = connected_labels==label # Make mask of the desired group
        volume_final[group_mask] = angle_class # Set angle
        

    # Save infos
    if slice_shape[0]==1:
        pl.plot_classified_map(volume_slice, file_path)
        pl.plot_classified_map(volume_final, file_path+"_volume_final")
    else:
        pl.Plot_Classified_Domain(volume_slice, file_path, remove_value=[1], labels=labels)
        pl.Plot_Classified_Domain(volume_final, file_path+"_volume_final", remove_value=[1], labels=labels)
        
        
    volume_final = volume_final.astype(np.uint8)
    volume_final.tofile(file_path + "_volume_final.raw")
    
    
    break
        

