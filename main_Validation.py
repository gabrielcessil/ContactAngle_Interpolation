import numpy as np
import Plotter as pl 
import utilities as util
import time


def load_volume(input_file_name, volume_shape, fluid_default_value):
    """Load the rock volume and return a binary volume representation."""
    volume_ground_truth = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
    volume_rock = (volume_ground_truth == fluid_default_value).astype(int)
    return volume_ground_truth, volume_rock

def load_measures(measure_file_name):
    """Load and preprocess the measurement data."""
    sampled_volume_info = np.load(measure_file_name) * np.pi / 180  # Remove this when fixed
    coordinates = sampled_volume_info[0:3, :]
    measures = sampled_volume_info[3, :]
    return coordinates, measures

def filter_valid_measures(measures):
    """Filter out invalid measurement values."""
    valid_mask = ~np.isnan(measures) & (measures != 0) & (measures != 180) & (measures != 179)  # Remove when fixed
    return valid_mask

def compute_surface_stats(surface_volume):
    """Compute statistics on the surface volume."""
    
    N_sample_cells = np.count_nonzero(valid_mask)
    N_surface_cells = np.count_nonzero(surface_volume != fluid_default_value)
    Sample_per_SurfaceArea = 100 * N_sample_cells / N_surface_cells
    
    print("N sample cells:", N_sample_cells)
    print("N surface cells:", N_surface_cells)
    print("Sample cells / surface cells %:", Sample_per_SurfaceArea)
    

def create_sampled_volume(volume_rock, coordinates, measures, valid_mask):
    """Assign measures to the volume cells."""
    sampled_volume = volume_rock.copy()
    coordinates = coordinates[:, valid_mask].astype(int)
    measures = util.value_2_LBPM_class((measures[valid_mask] * 180 / np.pi).astype(int), keep_values=())  # Remove when fixed
    i_indices, j_indices, k_indices = coordinates
    sampled_volume[i_indices, j_indices, k_indices] = measures
    return sampled_volume, coordinates

def create_guided_sampled_volume(volume_rock, volume_ground_truth, coordinates):
    """Create a guided sampled volume with original ground truth values at sampled locations."""
    guided_sampled_volume = volume_rock.copy()
    i_indices, j_indices, k_indices = coordinates
    guided_sampled_volume[i_indices, j_indices, k_indices] = volume_ground_truth[i_indices, j_indices, k_indices]
    return guided_sampled_volume




###############################################################################
#--- USER INPUTS --------------------------------------------------------------

# Method setup
experiment_id = "krig_test_8Neigh" # Give the experiment a base name
interpolation_mode = 'kriging' # Use one of: 'nn' 'watershed_grain' 'expand_samples'    
make_plots = True

# Domain setup
fluid_default_value= 1
solid_default_value= 0
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": 
        (
        "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__0_volume_final.raw", # rock .raw
        "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__0_measures/Angles_old_esfera.npy" # rock measures
        ),
}
#------------------------------------------------------------------------------
###############################################################################

output_base_folder_name = "Interpolated Volumes/"+interpolation_mode+"/"+experiment_id+"/"

# Perfomance metrics
mae = []
exec_time = []
for title, (input_file_name, measure_file_name) in input_files.items():            
    
    
    ###############################################################################
    #--- LOADING INFOS -------------------------------------------------------------
    # Load rock volume
    volume_ground_truth, volume_rock = load_volume(input_file_name, volume_shape, fluid_default_value)
    
    # Load measures
    coordinates, measures = load_measures(measure_file_name)
    
    # Filter valid measurements
    valid_mask = filter_valid_measures(measures)
    
    # Compute surface statistics
    print("Computing surface statistics")
    surface_volume = util.Remove_Internal_Solid(volume_rock)
    compute_surface_stats(surface_volume)
    
    # Assign measures to the volume
    sampled_volume, coordinates = create_sampled_volume(volume_rock, coordinates, measures, valid_mask)
    
    # Create guided sampled volume
    guided_sampled_volume = create_guided_sampled_volume(volume_rock, volume_ground_truth, coordinates)
    #------------------------------------------------------------------------------
    ###############################################################################        
    
    
    
    
    output_base_file_name = output_base_folder_name+title+"/"
    
    #sampled_volume = sampled_volume[0:100, 0:100, 0:100]
    #surface_volume = surface_volume[0:100, 0:100, 0:100]
    #volume_ground_truth = volume_ground_truth[0:100, 0:100, 0:100]
    #guided_sampled_volume = guided_sampled_volume[0:100, 0:100, 0:100]
    
    if make_plots:
        pl.Plot_Domain(sampled_volume, 
                        output_base_file_name+"sampled_volume", 
                        remove_value=[1],
                        special_colors = {
                            0: (0.5, 0.5, 0.5, 1.0),
                            1: (0.0, 0.0, 0.0, 1.0)
                        })
        
        pl.Plot_Domain(volume_ground_truth, 
                        output_base_file_name+"ground_truth_volume", 
                        remove_value=[1],
                        special_colors = {
                            0: (0.5, 0.5, 0.5, 1.0),
                            1: (0.0, 0.0, 0.0, 1.0)
                        })
    
        
    
    ###############################################################################
    #--- COMPUTATION --------------------------------------------------------------
    print("Starting computation")
    start_time = time.time()
    interpolated_volume = util.GET_INTERPOLATED_DOMAIN( sampled_volume, interpolation_mode )
    stopping_time = time.time()
    print("Ending Computation")
    
    pl.Plot_Domain(interpolated_volume, 
                   output_base_file_name+"interpolated", 
                   remove_value=[1],
                   special_colors = {
                       0: (0.5, 0.5, 0.5, 1.0),
                       1: (0.0, 0.0, 0.0, 1.0)
                   })
    
    #------------------------------------------------------------------------------
    ###############################################################################
    
    
    ###############################################################################
    #--- MAKING PLOTS -------------------------------------------------------
    
    if make_plots:
        print("Plotting:")
        # COMPUTATIONS FOR PLOTS 
        
        # Guided Interpolation: what the interpolation looks like if the samples are perfectly measured (but keeping location)
        guided_interpolated_volume = util.GET_INTERPOLATED_DOMAIN( guided_sampled_volume, interpolation_mode)
    
        surface_mask = surface_volume != fluid_default_value # True -> Interface cell, False -> Internal Solid or Fluid
        samples_mask = (sampled_volume != fluid_default_value) & (sampled_volume != solid_default_value) # True -> Samples cell, False - > Non sample cell 
        interpolated_mask = (interpolated_volume != fluid_default_value) & (interpolated_volume != solid_default_value)
        guided_interpolated_mask = (guided_interpolated_volume != fluid_default_value) & (guided_interpolated_volume != solid_default_value)
        sampled_surface_mask = (surface_mask) & (samples_mask)
        
        
        # SURFACE VALUES (surface_mask)
        ground_truth_surface_values = volume_ground_truth[surface_mask]
        ground_truth_surface_values_inDegrees = util.LBPM_class_2_value(ground_truth_surface_values, keep_values=())
        
        ground_truth_sampled_values = volume_ground_truth[sampled_surface_mask]
        ground_truth_sampled_values_inDegrees = util.LBPM_class_2_value(ground_truth_sampled_values, keep_values=())
        
        sampled_values = sampled_volume[samples_mask]
        sampled_values_inDegrees = util.LBPM_class_2_value(sampled_values, keep_values=())
        
        interpolated_surface_values = interpolated_volume[interpolated_mask]
        interpolated_surface_values_inDegrees = util.LBPM_class_2_value(interpolated_surface_values, keep_values=())
        
        guided_interpolated_surface_values = guided_interpolated_volume[guided_interpolated_mask]
        guided_interpolated_surface_values_inDegrees = util.LBPM_class_2_value(guided_interpolated_surface_values, keep_values=())
        
        
        # SURFACE ERRORS
        intp_and_gt_mask = (interpolated_mask) & (surface_mask) 
        cell = interpolated_volume[intp_and_gt_mask]
        ref = volume_ground_truth[intp_and_gt_mask]
        interpolation_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
        interpolation_error_volume_inDegrees = np.full(volume_shape, -1000)
        interpolation_error_volume_inDegrees[intp_and_gt_mask] = interpolation_error_inDegrees
        
        
        guidintp_and_gt_mask = (guided_interpolated_mask) & (surface_mask)
        cell = guided_interpolated_volume[guidintp_and_gt_mask]
        ref = volume_ground_truth[guidintp_and_gt_mask]
        guided_interpolation_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
        guided_interpolation_error_volume_inDegrees = np.full(volume_shape, -1000)
        guided_interpolation_error_volume_inDegrees[guidintp_and_gt_mask] = guided_interpolation_error_inDegrees
        
        
        guidance_and_gt_mask = (guided_interpolated_mask) & (interpolated_mask)
        cell = interpolated_volume[guidance_and_gt_mask]
        ref = guided_interpolated_volume[guidance_and_gt_mask]
        guidance_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
        guidance_error_volume_inDegrees = np.full(volume_shape, -1000)
        guidance_error_volume_inDegrees[guidance_and_gt_mask] = guidance_error_inDegrees
        
        
        # SAMPLES VALUES (samples_mask)
        samples_and_gt_mask = (samples_mask) & (surface_mask)
        cell = sampled_volume[samples_and_gt_mask]
        ref = volume_ground_truth[samples_and_gt_mask]
        sampled_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
        
        
        # Define the ground_truth contact angles for plotting
        ground_truth_classes = np.unique(volume_ground_truth[samples_and_gt_mask])
        ground_truth_classes_inDegrees = util.LBPM_class_2_value(ground_truth_classes)
        
        
        print("--Plotting Histograms")
        pl.plot_hist(
            {'Ground Truth': ground_truth_surface_values_inDegrees, 
             'Sampled Ground Truth':ground_truth_sampled_values_inDegrees,
             'Measures': sampled_values_inDegrees, # OK
             'Guided Interpolation': guided_interpolated_surface_values_inDegrees,
             'Interpolation': interpolated_surface_values_inDegrees,
            },
            bins=40, 
            title="Distribution Comparison",
            notable= ground_truth_classes_inDegrees,
            filename=output_base_file_name+"histogram",
            xlim=(0,180)
            )
             
        pl.plot_hist(
            {'Guided Interpolation errors (guided interpolation surface  -  ground truth surface)': guided_interpolation_error_inDegrees,
             'Measures errors (samples  -  ground_truth sampled cells)': sampled_error_inDegrees,
             'Interpolation errors (interpolation surface  -  ground truth surface)': interpolation_error_inDegrees, 
             'Guidance Error (interpolation surface - guided interpolation surface)': guidance_error_inDegrees
             },
            bins=40, 
            title="Distribution of Errors",
            filename=output_base_file_name+"error_histogram",
            xlim=(-180, 180)
            )
        
        print("--Plotting Domains")
        pl.Plot_Domain(interpolated_volume, 
                       output_base_file_name+"interpolated", 
                       remove_value=[1],
                       special_colors = {
                           0: (0.5, 0.5, 0.5, 1.0),
                           1: (0.0, 0.0, 0.0, 1.0)
                       })
        
        pl.Plot_Domain(guided_interpolated_volume, 
                       output_base_file_name+"guided_interpolated", 
                       remove_value=[1],
                       special_colors = {
                           0: (0.5, 0.5, 0.5, 1.0),
                           1: (0.0, 0.0, 0.0, 1.0)
                       })
        
        pl.Plot_Domain(interpolation_error_volume_inDegrees, 
                       output_base_file_name+"interpolation_error", 
                       remove_value=[-1000],
                       clim=[np.min(interpolation_error_volume_inDegrees), np.max(interpolation_error_volume_inDegrees)],
                       colormap='RdBu',
                       lbpm_class=False)
        
        pl.Plot_Domain(guided_interpolation_error_volume_inDegrees, 
                       output_base_file_name+"guided_interpolation_error", 
                       remove_value=[-1000],
                       clim=[np.min(guided_interpolation_error_volume_inDegrees), np.max(guided_interpolation_error_volume_inDegrees)],
                       colormap='RdBu',
                       lbpm_class=False)
        
        pl.Plot_Domain(guidance_error_volume_inDegrees, 
                       output_base_file_name+"guidance_error", 
                       remove_value=[-1000],
                       clim=[np.min(guidance_error_volume_inDegrees), np.max(guidance_error_volume_inDegrees)],
                       colormap='RdBu',
                       lbpm_class=False)

        
        
        # ANALYSIS PER CLASS
        colors = ['green', 'red', 'yellow']
        print()
        for gt_class, color in zip(ground_truth_classes,colors):
            
            class_mask = (volume_ground_truth == gt_class)
            
            intp_and_gt_mask = (interpolated_mask) & (surface_mask) & (class_mask)
            cell = interpolated_volume[intp_and_gt_mask]
            ref = volume_ground_truth[intp_and_gt_mask]
            interpolation_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
            interpolation_error_volume_inDegrees = np.full(volume_shape, -1000)
            interpolation_error_volume_inDegrees[intp_and_gt_mask] = interpolation_error_inDegrees
            
            
            guidintp_and_gt_mask = (guided_interpolated_mask) & (surface_mask)  & (class_mask)
            cell = guided_interpolated_volume[guidintp_and_gt_mask]
            ref = volume_ground_truth[guidintp_and_gt_mask]
            guided_interpolation_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
            guided_interpolation_error_volume_inDegrees = np.full(volume_shape, -1000)
            guided_interpolation_error_volume_inDegrees[guidintp_and_gt_mask] = guided_interpolation_error_inDegrees
            
            
            guidance_and_gt_mask = (guided_interpolated_mask) & (interpolated_mask) & (class_mask)
            cell = interpolated_volume[guidance_and_gt_mask]
            ref = guided_interpolated_volume[guidance_and_gt_mask]
            guidance_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
            guidance_error_volume_inDegrees = np.full(volume_shape, -1000)
            guidance_error_volume_inDegrees[guidance_and_gt_mask] = guidance_error_inDegrees
            
            
            samples_and_gt_mask = (samples_mask) & (surface_mask)  & (class_mask)
            cell = sampled_volume[samples_and_gt_mask]
            ref = volume_ground_truth[samples_and_gt_mask]
            sampled_error_inDegrees = util.LBPM_class_2_value(cell, keep_values=()) - util.LBPM_class_2_value(ref, keep_values=())
            
            pl.plot_hist(
                {'Guided Interpolation errors (guided interpolation surface  -  ground truth surface)': guided_interpolation_error_inDegrees,
                 'Measures errors (samples  -  ground_truth sampled cells)': sampled_error_inDegrees,
                 'Interpolation errors (interpolation surface  -  ground truth surface)': interpolation_error_inDegrees, 
                 'Guidance Error (interpolation surface - guided interpolation surface)': guidance_error_inDegrees
                 },
                bins=40, 
                title="Distribution of Errors for Contact Angle of {util.LBPM_class_2_value(gt_class)}",
                filename=output_base_file_name+f"error_histogram_{util.LBPM_class_2_value(gt_class)}",
                xlim=(-180, 180),
                color=color
                )
        
    #------------------------------------------------------------------------------
    ###############################################################################
    
    
    
    ###############################################################################
    #--- PERFORMANCE VALIDATION ---------------------------------------------------
    print("Getting metrics")
    metrics_info = util.Get_Metrics(volume_ground_truth, interpolated_volume, sampled_volume)
    mae.append(metrics_info['MAE'])    
    exec_time.append(stopping_time - start_time)
    #------------------------------------------------------------------------------
    ###############################################################################
    

print("\n\nFINAL METRICS: ")
print("MAE: ", np.average(mae), " +\- ", np.std(mae))
print("EXEC_TIME: ", np.average(exec_time), " +\- ", np.std(exec_time))
print("------\n\n\n")

    

         