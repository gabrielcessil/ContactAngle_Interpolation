import numpy as np
import Plotter as pl
import time
import utilities as util


###############################################################################
#--- USER INPUTS --------------------------------------------------------------
# The experiment identifier, give it the name you want
experiment_id = "exp_kriging"
# The interpolation method: 'expand_samples'(prefered) 'nn' 'watershed_grain' 
interpolation_mode = 'expand_samples'
# Number of uniformly distributed samples (repeated the process for each number of samples)
#N_samples = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2] #  2D
#N_samples = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.002, 0.003, 0.004, 0.005] #  3D
N_samples = [0.001] 
# Repeats the process within the same number of samples but in different surface points
samples_sets_trials = 1 
# Compute and save the plots
make_plots = True
#------------------------------------------------------------------------------
###############################################################################

#--- LOADING INFOS -------------------------------------------------------------
fluid_default_value= 1 # Fluid convention value found in the rock volume
solid_default_value= 0 # Solid convention value found in the rock volume

# 3D Volumes with wetability driven/clustered by Grains
"""
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__0_volume_final.raw",
    "Bentheimer_1": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__1_volume_final.raw",
    "Bentheimer_2": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__2_volume_final.raw",
    "Bentheimer_3": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__3_volume_final.raw",
    "Bentheimer_4": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__4_volume_final.raw",
    "Bentheimer_5": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__5_volume_final.raw",
    "Bentheimer_6": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__6_volume_final.raw",
    "Bentheimer_7": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__7_volume_final.raw",
    "Bentheimer_8": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__8_volume_final.raw",
    "Bentheimer_9": "Rock Volumes/Bentheimer/grains/benthheimer_200x200x200__9_volume_final.raw",
}
"""

# 3D Volumes with wetability driven/clustered by Pores
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__0_volume_final.raw",
    "Bentheimer_1": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__1_volume_final.raw",
    "Bentheimer_2": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__2_volume_final.raw",
    "Bentheimer_3": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__3_volume_final.raw",
    "Bentheimer_4": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__4_volume_final.raw",
    "Bentheimer_5": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__5_volume_final.raw",
    "Bentheimer_6": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__6_volume_final.raw",
    "Bentheimer_7": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__7_volume_final.raw",
    "Bentheimer_8": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__8_volume_final.raw",
    "Bentheimer_9": "Rock Volumes/Bentheimer/pores/benthheimer_200x200x200__9_volume_final.raw",
}

# 2D Volume example
"""
volume_shape = (1,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/Bentheimer/pores/benthheimer_1x200x200__2_volume_final.raw",
}
"""
#------------------------------------------------------------------------------
###############################################################################

#--- COMPUTATIONS -------------------------------------------------------------
output_base_folder_name = "Interpolated Volumes/"+"/"+experiment_id+"/"

infos_N = {}

for N in N_samples:
    print(f"Analysing N={N}: ")
    
    infos_N[N] = {}
    mae = []
    acc = []
    iou = []
    exec_time = []
        
    for title, input_file_name in input_files.items():            
        
        
        ###############################################################################
        #--- LOADING INFOS -------------------------------------------------------------
        print("Loading file: ", input_file_name)
        volume_ground_truth = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
        #------------------------------------------------------------------------------
        ###############################################################################
                    
        
        output_base_file_name = output_base_folder_name+title+f"/{N}_"
        if make_plots:
            print("Plotting file: ",  output_base_folder_name+title+"/"+"ground_truth_volume")
            if volume_shape[0]==1:
                pl.plot_classified_map(volume_ground_truth,  output_base_folder_name+title+"/"+"ground_truth_volume")
            else:
                pl.Plot_Domain(volume_ground_truth, output_base_folder_name+title+"/"+"ground_truth_volume", remove_value=[1],special_labels = {
                    0: (0.5, 0.5, 0.5, 1.0),
                    1: (0.0, 0.0, 0.0, 1.0)
                })
            print("Plotted")
        for trial in range(samples_sets_trials):
            print("Samples set trial: ", trial)
        
            ###############################################################################
            #--- KEEPING N RANDOM SAMPLES: SUBSTITUTE BY ANGULAR SAMPLING ALGORITHM -------
            # From validation rocks, collect random samples and let others be solid
            sampled_volume = util.Keep_random_samples(
                volume_ground_truth,
                N=N,
                solid_value=solid_default_value,
                fluid_value=fluid_default_value)
            
            
            if make_plots:
                print("Plotting: ", output_base_folder_name+title+"/"+"sampled_volume")
                if volume_shape[0]==1:
                    pl.plot_classified_map(sampled_volume,  output_base_folder_name+title+"/"+"sampled_volume")
                else:
                    pl.Plot_Classified_Domain(sampled_volume, output_base_folder_name+title+"/"+"sampled_volume", remove_value=[1])
            #------------------------------------------------------------------------------
            ###############################################################################
            
            
            
            ###############################################################################
            #--- INTERPOLATION ------------------------------------------------------------
            print("Starting computation")
            start_time = time.time()
            
            interpolated_volume = util.GET_INTERPOLATED_DOMAIN(sampled_volume, interpolation_mode)
            
            if make_plots:
                if volume_shape[0]==1:
                    pl.plot_classified_map(interpolated_volume,  output_base_file_name+"/"+"interpolated")
                else:
                    pl.Plot_Classified_Domain(interpolated_volume, output_base_file_name+"/"+"interpolated", remove_value=[1])
            
            stopping_time = time.time()
            print("Ending Computation")
            #------------------------------------------------------------------------------
            ###############################################################################
            
            
            
            ###############################################################################
            #--- VALIDATION PERFORMANCE --------------------------------------------------------------
            print("Getting metrics")
            metrics_info = util.Get_Metrics(volume_ground_truth, interpolated_volume, sampled_volume)
            mae.append(metrics_info['MAE'])
            acc.append(metrics_info["Accuracy"])
            iou.append(metrics_info['IOU'])       
            exec_time.append(stopping_time - start_time)
            ###############################################################################
        
        
    infos_N[N]["mae_avg"] = np.average(mae)
    infos_N[N]["mae_std"] = np.std(mae)
    
    infos_N[N]["acc_avg"] = np.average(acc)
    infos_N[N]["acc_std"] = np.std(acc)
    
    infos_N[N]["iou_avg"] = np.average(iou)
    infos_N[N]["iou_std"] = np.std(iou)
    
    infos_N[N]["time_avg"] = np.average(exec_time)
    infos_N[N]["time_std"] = np.std(exec_time)
    
    
    
    print("\n\nFINAL METRICS: ")
    print("MAE: ", np.average(mae), " +\- ", np.std(mae))
    print("Accuracy: ", np.average(acc), " +\- ", np.std(acc))
    print("IOU: ", np.average(iou), " +\- ", np.std(iou))
    print("EXEC_TIME: ", np.average(exec_time), " +\- ", np.std(exec_time))
    print("------\n\n\n")
        
        
            
mae_avg = [infos_N[N]["mae_avg"] for N in N_samples]
mae_std = [infos_N[N]["mae_std"] for N in N_samples]
samples_percent = np.array(N_samples)*100
pl.plot_mean_deviation(samples_percent, mae_avg, mae_std, 
                    title="Algorithm Performance",
                    xlabel="(Measured Cells / Surface Cells) % ", 
                    ylabel="Mean Absolute Error from Samples to Interpolated cells",
                    filename=output_base_folder_name+"mae")


acc_avg = [infos_N[N]["acc_avg"] for N in N_samples]
acc_dev = [infos_N[N]["acc_std"] for N in N_samples]
pl.plot_mean_deviation(samples_percent, acc_avg, acc_dev, 
                    title="Algorithm Performance",
                    xlabel="(Measured Cells / Surface Cells) % ", 
                    ylabel="Accuracy",
                    filename=output_base_folder_name+"acc")


time_avg = [infos_N[N]["time_avg"] for N in N_samples]
time_std = [infos_N[N]["time_std"] for N in N_samples]
pl.plot_mean_deviation(samples_percent, time_avg, time_std, 
                    title="Algorithm Performance",
                    xlabel="(Measured Cells / Surface Cells) % ", 
                    ylabel="Execution Time",
                    filename=output_base_folder_name+"exec_time")