import numpy as np
import Plotter as pl 
import utilities as util
import time
import matplotlib.pyplot as plt

def load_ground_truth_volume(input_file_name, volume_shape, fluid_default_value, solid_default_value):
    """Load the rock volume and return a binary volume representation."""
    volume_ground_truth = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
    
    if not np.all((volume_ground_truth == solid_default_value) | (volume_ground_truth == fluid_default_value) | util.is_LBPM_class(volume_ground_truth) ): 
        raise Exception(f"{input_file_name} values are outside LBPM unique classe's range: (0, 1, 74..254)")
        
    volume_rock = (volume_ground_truth == fluid_default_value).astype(int)
    return volume_ground_truth, volume_rock

def load_measures(measure_file_name, measures_format):
    """Load and preprocess the measurement data."""
    Measurements = np.load(measure_file_name) 
    
    angles      = Measurements[0,:]
    coordinates = Measurements[1:4,:].astype(int)
    
    # Turn rad measures into degrees 
    if measures_format=='rad': deg_measures = angles * (180/np.pi)
    else: deg_measures = angles
    
    return coordinates, deg_measures  

def filter_valid_measures(measures_deg):
    """Filter out invalid measurement values."""
    valid_mask = ~np.isnan(measures_deg) & (measures_deg >= 0) & (measures_deg <= 180)
    return valid_mask

def display_surface_stats(surface_volume):
    """Compute statistics on the surface volume."""
    
    N_sample_cells = np.count_nonzero(valid_mask)
    N_surface_cells = np.count_nonzero(surface_volume != fluid_default_value)
    Sample_per_SurfaceArea = 100 * N_sample_cells / N_surface_cells
    
    print(" -- N sample cells:", N_sample_cells)
    print(" -- N surface cells:", N_surface_cells)
    print(" -- Sample cells / surface cells %:", Sample_per_SurfaceArea)
    

def create_sampled_volume(volume_rock, coordinates, measures_deg, valid_mask):
    """Assign measures to the volume cells, letting others as default solid"""
    sampled_volume  = volume_rock.copy()
    coordinates     = coordinates[:, valid_mask].astype(int)
    measures_class  = util.value_2_LBPM_class((measures_deg[valid_mask]).astype(int), keep_values=())
    i_indices, j_indices, k_indices                 = coordinates
    sampled_volume[i_indices, j_indices, k_indices] = measures_class 
    return sampled_volume, coordinates

from collections import Counter
def create_guided_sampled_volume(volume_rock, volume_ground_truth, coordinates):
    """Create a guided sampled volume with original ground truth values at sampled locations."""
    guided_sampled_volume = volume_rock.copy()
    #i_indices, j_indices, k_indices = coordinates
    for i_indices, j_indices, k_indices in zip(*coordinates):
        near_ground_truth_values = util.Get_Neighbors(volume_ground_truth, i_indices, j_indices, k_indices)
        filtered= [v for v in near_ground_truth_values if v not in (0, 1)]
        if filtered:
            most_common_value, count = Counter(filtered).most_common(1)[0]
            guided_sampled_volume[i_indices, j_indices, k_indices] = most_common_value
    return guided_sampled_volume




###############################################################################
#--- USER INPUTS --------------------------------------------------------------

# Method setup
experiment_id       = "SampledDistribution" # Give the experiment a base name
interpolation_mode  = 'expand_samples' # Use one of: 'nn' 'watershed_grain' 'expand_samples'    
make_plots          = True 

# Domain convention setup
fluid_default_value = 1 # Default values for fluid/void cells
solid_default_value = 0 # Default values for solid cells            
volume_shape        = (200,200,200) 

# Input files in format: {"desired title for the rock": ("path to rock .raw", "path for rock measures")}
# :: Ground Truth: must be LBPM Class (0, 1, 74...254)
# :: Measures must be rads or grad
input_files = { 
    """
    "Bentheimer_0": 
        (
        "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Slide Images/Exemplo Medicoes/benthheimer_200x200x200__0_volume_final.raw",                      # Ground Truth .raw
        "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Slide Images/Exemplo Medicoes/benthheimer_200x200x200__0_morpho_drain_it1mi_AngleMeasures_filtered.npy"   # Measures .npy
        ),
    """
}
measures_format = 'deg' # 'deg' or 'rad'

# Color labels (Domain plot) in format {cell value: (alpha, Red, Green, Blue)}
special_colors = {
    solid_default_value: (0.5, 0.5, 0.5, 1.0), # Solid cells in GREY 
    fluid_default_value: (0.0, 0.0, 0.0, 1.0)  # Void cells in BLACK
}
labels = {  0: "Original Rock",
            1: "Void Space",
            30: "Water-Wetting Cells (30º)",
            90: "Neutral-Wetting Cells (90º)",
            150: "Oil-Wetting Cells (150º)"
}

#------------------------------------------------------------------------------
###############################################################################

output_base_folder_name = "Interpolated Volumes/"+experiment_id+"/"
labels = {util.value_2_LBPM_class(angle): label for angle, label in labels.items()}


# Perfomance metrics
accuracy = []
guided_accuracy = []
sampled_percent = []
# Perform analysis for for each rock listed
for title, (input_file_name, measure_file_name) in input_files.items():            
    
    ###############################################################################
    #--- LOADING INFOS -------------------------------------------------------------
    # Load rock volume (in LBPM class)
    volume_ground_truth, volume_rock = load_ground_truth_volume(input_file_name, volume_shape, fluid_default_value, solid_default_value)
    # Load measures (return in degrees)
    coordinates, measures_deg = load_measures(measure_file_name, measures_format)
    # Assert dimensions x, y and z
    assert volume_shape[0] >= max(coordinates[0]) and min(coordinates[0]) >=0, "Make sure the coordinates match the ground truth volume. "
    assert volume_shape[1] >= max(coordinates[1]) and min(coordinates[1]) >=0, "Make sure the coordinates match the ground truth volume. "
    assert volume_shape[2] >= max(coordinates[2]) and min(coordinates[2]) >=0, "Make sure the coordinates match the ground truth volume. "
    # Filter valid measurements (locate measures in proper range)
    valid_mask      = filter_valid_measures(measures_deg)
    #------------------------------------------------------------------------------
    ###############################################################################   



    ###############################################################################
    #--- TURN MEASURES INTO CLASSES            ------------------------------------
    ideal_centroids     = np.unique(volume_ground_truth)
    ideal_centroids     = ideal_centroids[~np.isin(ideal_centroids, [fluid_default_value, solid_default_value])] # Remove solid and void classes
    ideal_centroids_deg = util.LBPM_class_2_value(ideal_centroids)
    # Set measures to class (cluster centroids)
    measures_deg = util.get_to_GaussianMixture_label(measures_deg, ideal_centroids_deg, fluid_default_value=1, solid_default_value=0)
    # Compute surface statistics
    print("Computing surface statistics")
    surface_volume = util.Remove_Internal_Solid(volume_rock, fluid_default_value)
    display_surface_stats(surface_volume)
    # Assign valid measures to a solid volume (in LBPM class)
    print("Distributing measures in the surface")
    sampled_volume, valid_coordinates = create_sampled_volume(volume_rock, coordinates, measures_deg, valid_mask)
    #------------------------------------------------------------------------------
    ###############################################################################        
    
    
    

    ###############################################################################
    #--- COMPUTATION --------------------------------------------------------------
    print("\nVolume interpolation: Starting computation (counting time)")
    print("sampled volume: ", np.max(sampled_volume), np.min(sampled_volume))
    start_time = time.time()
    interpolated_volume = util.GET_INTERPOLATED_DOMAIN( sampled_volume, interpolation_mode, fluid_default_value, solid_default_value)
    stopping_time = time.time()
    print("Volume interpolation: Ending Computation")
    print("Volume GUIDED interpolation: Starting computation (not counting time)")
    # Guided Interpolation: what the interpolation looks like if the samples are perfectly measured (but keeping location)
    guided_sampled_volume = create_guided_sampled_volume(volume_rock, volume_ground_truth, valid_coordinates)
    guided_interpolated_volume = util.GET_INTERPOLATED_DOMAIN( guided_sampled_volume, interpolation_mode, fluid_default_value, solid_default_value)
    print("Volume GUIDED interpolation: Ending Computation") 
    #------------------------------------------------------------------------------
    ###############################################################################   
    
    output_base_file_name = output_base_folder_name+title+"/"
    
    ###############################################################################
    #--- MAKING PLOTS --------------------------------------------------------------
    if make_plots:
        print("Making plots")
        
        pl.Plot_Domain(sampled_volume, 
                        output_base_file_name+"sampled_volume", 
                        remove_value=[fluid_default_value],
                        special_colors= {
                            0: (0.5, 0.5, 0.5, 0.1), # Assign grey for solid cells
                            1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                        })
        pl.Plot_Classified_Domain(volume_ground_truth, 
                        output_base_file_name+"ground_truth_volume", 
                        remove_value=[fluid_default_value],
                        special_colors = special_colors,
                        labels=labels)
        pl.Plot_Classified_Domain(interpolated_volume, 
                        output_base_file_name+"interpolated_volume", 
                        remove_value=[fluid_default_value],
                        special_colors = special_colors,
                        labels=labels)
        pl.Plot_Classified_Domain(guided_interpolated_volume, 
                        output_base_file_name+"guided_interpolated_volume", 
                        remove_value=[fluid_default_value],
                        special_colors = special_colors,
                        labels=labels)
        pl.Plot_Classified_Domain(guided_sampled_volume, 
                        output_base_file_name+"guided_sampled_volume", 
                        remove_value=[fluid_default_value],
                        special_colors = special_colors,
                        labels=labels)
    #------------------------------------------------------------------------------
    ###############################################################################
    
    
    # Measure clustering affectiviness
    i_indices, j_indices, k_indices = valid_coordinates
    measures_ground_truth = guided_sampled_volume[i_indices, j_indices, k_indices]
    
    #------------------------------------------------------------------------------
    ###############################################################################




    ###############################################################################
    #--- SAVING -------------------------------------------------------
    # Resulting interpolation
    np.save(output_base_file_name+"result"+".npy", interpolated_volume)
    interpolated_volume.tofile(output_base_file_name+"result"+".raw")
    # Guided interpolation
    np.save(output_base_file_name+"result"+".npy", guided_interpolated_volume)
    guided_interpolated_volume.tofile(output_base_file_name+"result"+".raw")
    #------------------------------------------------------------------------------
    ###############################################################################
    
    
    
    
    ###############################################################################
    #--- PERFORMANCE VALIDATION ---------------------------------------------------
    print("\nGetting performance metrics")
    metrics_info        = util.Get_Metrics(volume_ground_truth, interpolated_volume, sampled_volume)
    accuracy.append(metrics_info['Accuracy'])
    guided_metrics_info = util.Get_Metrics(volume_ground_truth, guided_interpolated_volume, sampled_volume)
    guided_accuracy.append(guided_metrics_info['Accuracy'])
    sampled_percent.append(100*len(measures_deg)/np.count_nonzero(surface_volume==solid_default_value))
    #------------------------------------------------------------------------------
    ###############################################################################
    

###############################################################################
#--- SAVING PERFORMANCE STATISTICS --------------------------------------------
print("\n\nFINAL METRICS: ")
print("Sampled percentual: ",   np.average(sampled_percent), " +\- ", np.std(sampled_percent))
print("Accuracy: ",             np.average(accuracy), " +\- ", np.std(accuracy))
print("Guided Accuracy: ",      np.average(guided_accuracy), " +\- ", np.std(guided_accuracy))
print("------\n\n\n")
np.save(output_base_folder_name+"samples_percent.npy",  np.array(sampled_percent))
np.save(output_base_folder_name+"accuracy.npy",         np.array(accuracy))
#------------------------------------------------------------------------------
###############################################################################
    
