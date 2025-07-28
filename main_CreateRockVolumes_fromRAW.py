import numpy as np
import random
import Plotter as pl
import utilities as util
import os

def write_lbpm_db(path_name: str, raw_volume_filename: str):
    """
    Writes an LBPM-compatible .db configuration file with a custom raw volume filename.

    Parameters:
    - db_filename (str): Output .db file path (e.g., 'wet.db')
    - raw_volume_filename (str): Name of the .raw file to be referenced in the Domain block
    """
    db_content = f'''Domain {{
   Filename    = "{raw_volume_filename}"
   ReadType    = "8bit"    // data type
   N           = 200, 200, 200       // size of original image
   nproc       = 2, 2, 1       // process grid
   n           = 100, 100, 200         // sub-domain size
   voxel_length   = 1.0    // voxel length (in microns)
   ReadValues     = 0, 1, 2, 119, 209  // labels within the original image
   WriteValues    = 0, 1, 2, -1, -2  // associated labels to be used by LBPM
   BC             = 0                // fully periodic BC
}}

Color {{
   protocol          = "fractional flow"
   capillary_number  = 1e-4           // capillary number for the displacement, positive="oil injection"
   timestepMax       = 1000000               // maximum timtestep
   alpha = 0.01                       // controls interfacial tension
   rhoA  = 1.0                         // controls the density of fluid A
   rhoB  = 1.0                         // controls the density of fluid B
   tauA  = 0.7                         // controls the viscosity of fluid A
   tauB  = 0.7                         // controls the viscosity of fluid B
   F     = 0, 0, 0                     // body force
   WettingConvention = "SCAL"
   ComponentLabels   = 0, -1, -2        // image labels for solid voxels
   ComponentAffinity = 0, -0.5, 0.5   // controls the wetting affinity for each label
   Restart           = false
}}

Analysis {{
   analysis_interval             = 50000  // logging interval for timelog.csv
   subphase_analysis_interval    = 50000  // loggging interval for subphase.csv
   N_threads                     = 0                          // number of analysis threads (GPU version only)
   visualization_interval        = 200000         // interval to write visualization files
   restart_interval              = 10000000            // interval to write restart file
   restart_file                  = "Restart"               // base name of restart file
}}

Visualization {{
   write_silo        = true   // write SILO databases with assigned variables
   save_8bit_raw     = true   // write labeled 8-bit binary files with phase assignments
   save_phase_field  = true  // save phase field within SILO database
   save_pressure     = true  // save pressure field within SILO database
   save_velocity     = true  // save velocity field within SILO database
}}

FlowAdaptor {{
   min_steady_timesteps = 10000000     // minimum number of timesteps per steady point
   max_steady_timesteps = 11000000     // minimum number of timesteps per steady point
   fractional_flow_increment = 0.0
}}
'''

    with open(path_name, "w") as f:
        f.write(db_content)

def write_slurm_script(path_name: str):
    """
    Writes a SLURM batch submission script for LBPM simulation to the specified filename.
    
    Parameters:
    - path_name (str): The path where the .sh script should be saved (e.g., 'submit.sh')
    """
    script_content = """#!/bin/bash

# ---------------- SLURM Job Settings ----------------

#SBATCH --job-name=wetting_interpore       # Job name for identification
#SBATCH --partition=all_gpu                # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --nodes=1                          # Number of nodes
##SBATCH --gres=gpu:k40m:4                 # (Commented) Request 4 K40m GPUs (uncomment to use)
#SBATCH -t 4-0:00                          # Max wall time: 4 days
#SBATCH -o slurm.%j.out                   # File to write standard output (%j = job ID)
#SBATCH -e slurm.%j.err                   # File to write standard error (%j = job ID)
#SBATCH --ntasks-per-node=4                # Number of tasks (processes) per node

# ---------------- Environment Setup ----------------

# Load the appropriate module (with CUDA-aware MPI already built)
module load lbpm/gpu/a114b5f7

# ---------------- Job Execution --------------------

# Run the simulation using MPI with 4 processes
mpirun -np 4 lbpm_color_simulator wet.db
"""
    with open(path_name, "w") as f:
        f.write(script_content)
        
        
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

input_file_name = r"Rock Volumes/bentheimer_900_900_1600.raw"
volume_shape    = (1600, 900, 900)
volume          = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)


fluid_default       = 1
solid_default       = 0

# Setting Grain(solid) = 0, Fluid(void) = 1
volume[volume==1]   = fluid_default
volume[volume==2]   = fluid_default
volume[volume==255] = solid_default
volume[volume==254] = solid_default

N                   = 30
slice_shape         = (200, 200, 200)

cluster_mode        = 'pores' # 'grains' or 'pores'

angles              = [45, 135]
probabilities       = [0.5, 0.5]

labels              = { 0:      "Original Solid Cells",
                        1:      "Void Space Cells",
                        45:     "Water-Wetting Cells (45º)",
                        135:    "Oil-Wetting Cells (135º)",}

special_colors      = {
    0: (0.5, 0.5, 0.5, 1.0), # Assign grey for solid cells
    1: (0.0, 0.0, 0.0, 1.0)  # Assign black for void cells (removed from plot)
}




# Converting labels values to LBPM class
labels          = {util.value_2_LBPM_class(angle): label for angle, label in labels.items()}
special_colors  = {util.value_2_LBPM_class(angle): color for angle, color in special_colors.items()}

random.seed(42)  # Use any integer you like; 42 is a common example

slices = extract_non_overlapping_slices(volume, slice_shape)
slices = random.sample(slices, N)

for i, volume_slice in enumerate(slices):
    
    print("Creating domain ", i)
    volume_final = volume_slice.copy()
    s_x, s_y, s_z = volume_slice.shape
    rock_name = input_file_name.replace(".raw", "")
    angles_str = "_" + "_".join(str(v) for v in angles)
    file_path = f"{rock_name}_{cluster_mode}_distribution/multiWet_rock_Ang{angles_str}_Shape{s_x}x{s_y}x{s_z}__{i}"
    
    if cluster_mode == 'grains':
        volume_to_cluster = (volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
        
        filter_cube = np.maximum(np.array(slice_shape)/4,1)
        connected_labels, valid_labels = filter_connected_labels(connected_labels, valid_labels, filter_cube.astype(int))
        
        """
        pl.Plot_Classified_Domain(
            connected_labels, 
            file_path+"_clusters", 
            remove_value=[np.min(connected_labels)],
            labels=labels,
            colormap='hsv'
        )
        """
        
            
    elif cluster_mode == 'pores':
        volume_to_cluster = fluid_default-(volume_slice != fluid_default).astype(np.uint8)
        connected_labels, valid_labels, distance_map, local_max = util.Watershed(volume_to_cluster)
         
        """
        pl.Plot_Classified_Domain(
        values=connected_labels, 
        filename = file_path+"_clusters_BeforeFiltering",
        show_label=False,
        special_colors=special_colors,
        colormap='hsv')
        """
        filter_cube = np.maximum(np.array(slice_shape)/2,1)
        connected_labels, valid_labels = filter_connected_labels(connected_labels, valid_labels, filter_cube.astype(int))
        
        """
        pl.Plot_Classified_Domain(
            values=connected_labels, 
            filename = file_path+"_clusters_AfterFiltering",
            show_label=False,
            special_colors=special_colors,
            colormap='hsv'
            )
        """
        
        
        hollow_volume       = util.Remove_Internal_Solid(volume_slice, connectivity=3)
        connected_labels    = util.Set_Solid_to_Void_Label(hollow_volume, connected_labels, valid_labels, connectivity=3)
        
        non_surface_solid_mask  = (volume_slice != fluid_default) & (hollow_volume!=solid_default)
        fluid_mask              = volume_slice == fluid_default
        connected_labels[non_surface_solid_mask]    = min(valid_labels)-1  # Set the solids to a unvalid label, so it dont get in raffle
        connected_labels[fluid_mask]                = min(valid_labels)-2  # Set the fluid to a unvalid label, so it dont get in raffle
        
        """
        pl.Plot_Classified_Domain(
            values   = connected_labels, 
            filename = file_path+"_clusters_AfterFiltering_onSurface",
            show_label=False,
            remove_value=[min(valid_labels)-1],
            colormap='hsv',
            special_colors={min(valid_labels)-1: (0.5, 0.5, 0.5, 1.0),  min(valid_labels)-2: (0.0, 0.0, 0.0, 1.0)},
            )
        """
        
        """
        aux_local_max = volume_slice.copy()
        aux_local_max[local_max] = 10
        pl.Plot_Classified_Domain(
            values=aux_local_max, 
            filename = file_path+"_local_max",
            labels={1: "Void Space cells", 0: "Solid Cells", 10: "Distance Transform's Local Maximum"},
            show_label=True,
            colormap='prism',
            special_colors=special_colors
            )
        pl.Plot_Classified_Domain(
            values=volume_slice, 
            filename = file_path+"_volume",
            labels={fluid_default: "Void Space Cells", solid_default: "Solid Cells"},
            show_label=True,
            special_colors=special_colors
            )
        
        pl.Plot_Classified_Domain(
            values=distance_map, 
            filename = file_path+"_distance_map",
            special_colors={0: special_colors[0]},
            colormap='inferno_r',
            show_label=False,
            )
        """
            
    else:
        raise Exception("Not implemented")
    
        
    # Filter result
    print(" - Filtering results")
    
    
    # Raffle contact angles across clusters
    # Raffle contact angles across clusters
    for label in valid_labels:
        angle = random.choices(angles, weights=probabilities, k=1)[0]  # Choose angle with weights
        angle_class = util.value_2_LBPM_class(angle)
        group_mask  = connected_labels == label
        volume_final[group_mask] = angle_class

        
    # Save plots
    
    print(" - Plotting")
    if slice_shape[0]==1:
        
        pl.Plot_Classified_Domain_2D(
            values=volume_final, 
            filename = file_path+"_volume_final",
            labels=labels,
            show_label=True,
            special_colors=special_colors
            )
    
        
    else:
        pl.Plot_Classified_Domain(
            volume_final, 
            file_path+"_volume_final", 
            remove_value=[fluid_default],
            labels=labels,
            special_colors=special_colors
        )
    
        
    # Save volume 
    print(" - Saving")
    if not os.path.exists(file_path+"/"): os.makedirs(file_path+"/")
    volume_final = volume_final.astype(np.uint8)
    volume_final.tofile(file_path + "/volume_withAngles.raw")
    volume_slice.tofile(file_path + "/volume.raw")
    write_lbpm_db(path_name = file_path+"/wet.db", raw_volume_filename = "volume_withAngles.raw")
    write_slurm_script(path_name = file_path+"/Run_sim.sh")