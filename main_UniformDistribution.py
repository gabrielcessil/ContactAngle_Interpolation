import numpy as np
import Plotter as pl
import time
import utilities as util


#######################################################
#************ INPUTS                       ***********#
#######################################################

# The experiment identifier, give it the name you want
experiment_id       = "expand_samples_pores_003"
interpolation_mode  = 'expand_samples'
# Number of uniformly distributed samples (repeated the process for each number of samples)
kept_fractions = [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] #  3D
samples_sets_trials = 1  # Repeats the process within the same number of samples but in different surface points
# Compute and save the plots
make_plots = False
labels              = { 0:      "Original Solid Cells",
                        1:      "Void Space Cells",
                        45:     "Water-Wetting Cells (45º)",
                        135:    "Oil-Wetting Cells (135º)",}

special_colors      = {
    0: (0.5, 0.5, 0.5, 1.0), # Assign grey for solid cells
    1: (0.0, 0.0, 0.0, 1.0)  # Assign black for void cells (removed from plot)
}

fluid_default_value= 1 # Fluid convention value found in the rock volume
solid_default_value= 0 # Solid convention value found in the rock volume

# 3D Volumes with wetability driven/clustered by Grains
"""
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__0_volume_withAngles.raw",
    "Bentheimer_1": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__1_volume_withAngles.raw",
    "Bentheimer_2": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__2_volume_withAngles.raw",
    "Bentheimer_3": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__3_volume_withAngles.raw",
    "Bentheimer_4": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__4_volume_withAngles.raw",
    "Bentheimer_5": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__5_volume_withAngles.raw",
    "Bentheimer_6": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__6_volume_withAngles.raw",
    "Bentheimer_7": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__7_volume_withAngles.raw",
    "Bentheimer_8": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__8_volume_withAngles.raw",
    "Bentheimer_9": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__9_volume_withAngles.raw",
    
    "Bentheimer_10": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__10_volume_withAngles.raw",
    "Bentheimer_11": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__11_volume_withAngles.raw",
    "Bentheimer_12": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__12_volume_withAngles.raw",
    "Bentheimer_13": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__13_volume_withAngles.raw",
    "Bentheimer_14": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__14_volume_withAngles.raw",
    "Bentheimer_15": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__15_volume_withAngles.raw",
    "Bentheimer_16": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__16_volume_withAngles.raw",
    "Bentheimer_17": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__17_volume_withAngles.raw",
    "Bentheimer_18": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__18_volume_withAngles.raw",
    "Bentheimer_19": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__19_volume_withAngles.raw",
    
    "Bentheimer_20": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__20_volume_withAngles.raw",
    "Bentheimer_21": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__21_volume_withAngles.raw",
    "Bentheimer_22": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__22_volume_withAngles.raw",
    "Bentheimer_23": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__23_volume_withAngles.raw",
    "Bentheimer_24": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__24_volume_withAngles.raw",
    "Bentheimer_25": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__25_volume_withAngles.raw",
    "Bentheimer_26": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__26_volume_withAngles.raw",
    "Bentheimer_27": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__27_volume_withAngles.raw",
    "Bentheimer_28": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__28_volume_withAngles.raw",
    "Bentheimer_29": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__29_volume_withAngles.raw",
}
"""

# 3D Volumes with wetability driven/clustered by Pores
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__0_volume_withAngles.raw",
    "Bentheimer_1": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__1_volume_withAngles.raw",
    "Bentheimer_2": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__2_volume_withAngles.raw",
    "Bentheimer_3": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__3_volume_withAngles.raw",
    "Bentheimer_4": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__4_volume_withAngles.raw",
    "Bentheimer_5": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__5_volume_withAngles.raw",
    "Bentheimer_6": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__6_volume_withAngles.raw",
    "Bentheimer_7": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__7_volume_withAngles.raw",
    "Bentheimer_8": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__8_volume_withAngles.raw",
    "Bentheimer_9": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__9_volume_withAngles.raw",
    
    "Bentheimer_10": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__10_volume_withAngles.raw",
    "Bentheimer_11": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__11_volume_withAngles.raw",
    "Bentheimer_12": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__12_volume_withAngles.raw",
    "Bentheimer_13": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__13_volume_withAngles.raw",
    "Bentheimer_14": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__14_volume_withAngles.raw",
    "Bentheimer_15": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__15_volume_withAngles.raw",
    "Bentheimer_16": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__16_volume_withAngles.raw",
    "Bentheimer_17": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__17_volume_withAngles.raw",
    "Bentheimer_18": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__18_volume_withAngles.raw",
    "Bentheimer_19": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__19_volume_withAngles.raw",
    
    "Bentheimer_20": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__20_volume_withAngles.raw",
    "Bentheimer_21": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__21_volume_withAngles.raw",
    "Bentheimer_22": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__22_volume_withAngles.raw",
    "Bentheimer_23": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__23_volume_withAngles.raw",
    "Bentheimer_24": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__24_volume_withAngles.raw",
    "Bentheimer_25": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__25_volume_withAngles.raw",
    "Bentheimer_26": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__26_volume_withAngles.raw",
    "Bentheimer_27": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__27_volume_withAngles.raw",
    "Bentheimer_28": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__28_volume_withAngles.raw",
    "Bentheimer_29": "Rock Volumes/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__29_volume_withAngles.raw",
}

# 3D Volume example
"""
volume_shape = (200,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__0_volume_withAngles.raw",
}
"""

# 2D Volume example
"""
volume_shape = (1,200,200)
input_files = {
    "Bentheimer_0": "Rock Volumes/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape1x200x200__11_volume_withAngles.raw"
}
"""



#######################################################
#************ COMPUTATION                  ***********#
#######################################################
"""

# Converting labels values to LBPM class
labels          = {util.value_2_LBPM_class(angle): label for angle, label in labels.items()}
special_colors  = {util.value_2_LBPM_class(angle): color for angle, color in special_colors.items()}

output_base_folder_name = f"Interpolated Volumes/Uniform_Distribution/{experiment_id}/"

infos_N = {}
for kept_fraction in kept_fractions:
    print(f"Analysing kept_fraction={kept_fraction}: ")
    
    infos_N[kept_fraction] = {}
    mae = []
    acc = []
    iou = []
    exec_time = []
        
    for title, input_file_name in input_files.items():            
        
        
        ###############################################################################
        #--- LOADING INFOS -------------------------------------------------------------
        print(" - Loading file: ", input_file_name)
        volume_ground_truth = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
        #------------------------------------------------------------------------------
        ###############################################################################
                    
        
        output_base_file_name = output_base_folder_name+title+f"/{kept_fraction}_"
        if make_plots:
            print("Plotting file: ",  output_base_folder_name+title+"/"+"ground_truth_volume")
            if volume_shape[0]==1:
                pl.Plot_Classified_Domain_2D(volume_ground_truth,  output_base_folder_name+title+"/"+"ground_truth_volume",special_colors=special_colors, labels=labels)

            else:
                pl.Plot_Classified_Domain(volume_ground_truth, output_base_folder_name+title+"/"+"ground_truth_volume", 
                                          remove_value=[1],
                                          special_colors=special_colors,
                                          labels=labels,
                                          )
                hollow_ground_truth = util.Remove_Internal_Solid(volume_ground_truth, connectivity=3)
                pl.Plot_Classified_Domain(hollow_ground_truth, output_base_folder_name+title+"/"+"ground_truth_volume_internal", 
                                          remove_value=[1],
                                          special_colors=special_colors,
                                          labels=labels)
                
                
                
                
                
            print("Plotted")
        for trial in range(samples_sets_trials):
            print(" --- Samples set trial: ", trial)
        
            ###############################################################################
            #--- KEEPING N RANDOM SAMPLES: SUBSTITUTE BY ANGULAR SAMPLING ALGORITHM -------
            # From validation rocks, collect random samples and let others be solid
            sampled_volume = util.Keep_random_samples(
                volume_ground_truth,
                kept_fraction=kept_fraction,
                solid_value=solid_default_value,
                fluid_value=fluid_default_value)
            
            
            
            if make_plots:
                print("Plotting: ", output_base_folder_name+title+"/"+"sampled_volume")
                if volume_shape[0]==1:
                    pl.Plot_Classified_Domain_2D(sampled_volume,  output_base_folder_name+title+"/"+"sampled_volume",special_colors=special_colors,labels=labels)
                else:
                    pl.Plot_Classified_Domain(sampled_volume, output_base_folder_name+title+"/"+"sampled_volume", 
                                              remove_value=[1],
                                              special_colors=special_colors,
                                              labels=labels)
            #------------------------------------------------------------------------------
            ###############################################################################
            
            
            
            ###############################################################################
            #--- INTERPOLATION ------------------------------------------------------------
            print(" --- Starting computation")
            start_time = time.time()
            
            interpolated_volume = util.GET_INTERPOLATED_DOMAIN(sampled_volume, interpolation_mode)
            
            if make_plots:
                if volume_shape[0]==1:
                    pl.Plot_Classified_Domain_2D(interpolated_volume,  output_base_file_name+"/"+"interpolated", special_colors=special_colors,labels=labels)

                else:
                    pl.Plot_Classified_Domain(interpolated_volume, output_base_file_name+"/"+"interpolated", remove_value=[1], special_colors=special_colors, labels=labels)
            
            stopping_time = time.time()
            print(" --- Ending Computation")
            #------------------------------------------------------------------------------
            ###############################################################################
            
            
            
            ###############################################################################
            #--- SAVING CROP'S PERFORMANCE --------------------------------------------------------------
            print(" --- Getting metrics")
            metrics_info = util.Get_Metrics(volume_ground_truth, interpolated_volume, sampled_volume)
            mae.append(metrics_info['MAE'])
            acc.append(metrics_info["Accuracy"])
            iou.append(metrics_info['MIOU'])       
            exec_time.append(stopping_time - start_time)
            ###############################################################################
        
    
    ###############################################################################
    #--- SAVING PERFORMANCE STATISTICS ----------------------------------------------------------
    infos_N[kept_fraction]["mae_avg"] = np.average(mae)
    infos_N[kept_fraction]["mae_std"] = np.std(mae)
    
    infos_N[kept_fraction]["acc_avg"] = np.average(acc)
    infos_N[kept_fraction]["acc_std"] = np.std(acc)
    
    infos_N[kept_fraction]["iou_avg"] = np.average(iou)
    infos_N[kept_fraction]["iou_std"] = np.std(iou)
    
    infos_N[kept_fraction]["time_avg"] = np.average(exec_time)
    infos_N[kept_fraction]["time_std"] = np.std(exec_time)
    

    print("\n\n - FINAL METRICS: ")
    print(" --- MAE: ", np.average(mae), " +\- ", np.std(mae))
    print(" --- Accuracy: ", np.average(acc), " +\- ", np.std(acc))
    print(" --- IOU: ", np.average(iou), " +\- ", np.std(iou))
    print(" --- EXEC_TIME: ", np.average(exec_time), " +\- ", np.std(exec_time))
    print("------\n\n\n")
    
    ###############################################################################
        
print(" PLOTTING METRICS vs Nº of SAMPLES")
samples_percent = np.array(kept_fractions)*100
acc_avg = [infos_N[N]["acc_avg"] for N in kept_fractions]
acc_dev = [infos_N[N]["acc_std"] for N in kept_fractions]


np.save(output_base_folder_name+"samples_percent.npy", np.array(samples_percent))
np.save(output_base_folder_name+"accuracy_avg.npy", np.array(acc_avg))
np.save(output_base_folder_name+"accuracy_std.npy", np.array(acc_dev))
"""



#######################################################
#************ VISUALIZATION                  *********#
#######################################################

#"""
base_folder = "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Interpolated Volumes/Uniform_Distribution/"

samples_percent_1 = np.load(base_folder+"expand_samples_pores/samples_percent_exp1.npy")
acc_avg_1         = np.load(base_folder+"expand_samples_pores/accuracy_avg_exp1.npy")
acc_dev_1         = np.load(base_folder+"expand_samples_pores/accuracy_std_exp1.npy")
samples_percent_2 = np.load(base_folder+"expand_samples_pores/samples_percent_exp2.npy")
acc_avg_2         = np.load(base_folder+"expand_samples_pores/accuracy_avg_exp2.npy")
acc_dev_2         = np.load(base_folder+"expand_samples_pores/accuracy_std_exp2.npy")
samples_percent = np.concatenate((samples_percent_1, samples_percent_2))  # → array([1, 2, 3, 4, 5])
acc_avg         = np.concatenate((acc_avg_1, acc_avg_2))
acc_dev         = np.concatenate((acc_dev_1, acc_dev_2))
sorted_indices  = np.argsort(samples_percent)
samples_percent_pores = samples_percent[sorted_indices]
acc_avg_pores         = acc_avg[sorted_indices]
acc_dev_pores         = acc_dev[sorted_indices]

samples_percent_1 = np.load(base_folder+"expand_samples_grains/samples_percent_exp1.npy")
acc_avg_1         = np.load(base_folder+"expand_samples_grains/accuracy_avg_exp1.npy")
acc_dev_1         = np.load(base_folder+"expand_samples_grains/accuracy_std_exp1.npy")
samples_percent_2 = np.load(base_folder+"expand_samples_grains/samples_percent_exp2.npy")
acc_avg_2         = np.load(base_folder+"expand_samples_grains/accuracy_avg_exp2.npy")
acc_dev_2         = np.load(base_folder+"expand_samples_grains/accuracy_std_exp2.npy")
samples_percent = np.concatenate((samples_percent_1, samples_percent_2))  # → array([1, 2, 3, 4, 5])
acc_avg         = np.concatenate((acc_avg_1, acc_avg_2))
acc_dev         = np.concatenate((acc_dev_1, acc_dev_2))
sorted_indices  = np.argsort(samples_percent)
samples_percent_grains = samples_percent[sorted_indices]
acc_avg_grains         = acc_avg[sorted_indices]
acc_dev_grains         = acc_dev[sorted_indices]


result = acc_avg_grains[samples_percent_grains == 0.3]
print("Grains-based expected performance for 0.3%: ", result)  
result = acc_avg_pores[samples_percent_pores == 0.3]
print("Pores-based expected performance for 0.3%: ", result)  


pl.plot_multiple_mean_deviation(
    {"Pores-based": (samples_percent_pores, acc_avg_pores, acc_dev_pores),
     "Grains-based": (samples_percent_grains, acc_avg_grains, acc_dev_grains)},
    title="Algorithm Performance",
    xlabel="Measured Cells / Surface Cells [%]", 
    ylabel="Accuracy",
    filename=base_folder+"acc")
#"""