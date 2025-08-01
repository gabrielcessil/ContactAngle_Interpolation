import Plotter as pl
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import sys
import os

#######################################################
#************ LOAD MORPH DRAIN ARRAY  :    ***********#
#######################################################

# Morph drain array
# Grain-based simulated domains
"""
input_file_names = [
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__0/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__1/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__2/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__3/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__4/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__5/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__6/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__7/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__8/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_grains_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__9/id_t1000000.raw",
]
volume_shape        = (200,200,200)
"""

# Pore-based simulated domains
"""
input_file_names = [
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__0/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__1/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__2/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__3/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__4/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__5/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__6/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__7/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__8/id_t1000000.raw",
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/bentheimer_900_900_1600_pores_distribution/multiWet_rock_Ang_45_135_Shape200x200x200__9/id_t1000000.raw",
]
volume_shape        = (200,200,200)
"""

# Grain-based simulated domains for Validation (Flooding)
input_file_names = [
    "/home/gabriel/remote/hal/Interpore_Wettability_Rocks/multiWet_rock_Ang_45_135_Shape600x600x600__0/id_t400000.raw",
]
volume_shape        = (600,600,600)


# Routines to be executed
make_plots          = False
make_measurements   = True




#######################################################
#************ RUN MEASURING ALGORITHM:     ***********#
#######################################################

if make_measurements:
    for input_file_name in input_file_names:
        print("Processing ", input_file_name)
        morph_drain         = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
        morph_drain[(morph_drain != 1) & (morph_drain != 2)]          = 0 # Set any kind of solid to default solid
        
        ## Sanity check: THE LPBM RAW MATCHES THE GROUND TRUTH?
        orig_volume         = np.fromfile(os.path.dirname(input_file_name)+"/volume.raw", dtype=np.uint8).reshape(volume_shape)
        if np.mean(abs(morph_drain[orig_volume==0]))!=0: raise Exception("Simulation do not match ground truth volume.")
            
        
        # Load measuring method
        file_dir = '/home/gabriel/Desktop/Molhabilidade/Metodo Medida Cris/' # Folder of where the library file is located. 
        sys.path.append(file_dir)
        import Lib_ContactAngle as esf
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Measurement parameters
        R1                      = 5     # Plane regression subvolume radius
        alpha                   = 0.5   # Sphere regression subvbolume radius scaling factors
        Max_residual_plane      = 0.4
        Max_residual_sphere     = 0.5
        Min_points_plane        = 0.5
        Min_points_sphere       = 20 
        # Measure contact angles
        Measurements            = esf.MeasureAngle(morph_drain)
        Filtered_Measurements   = esf.FilterMeasurements(Measurements, R1 = R1, mpe = Min_points_sphere, mpp = Min_points_plane, maxerr_e = Max_residual_sphere, maxerr_p = Max_residual_plane)
            
        # Save results:
        np.save(os.path.dirname(input_file_name)+"/AngleMeasures.npy", Measurements[0:4, :])
        np.save(os.path.dirname(input_file_name)+"/AngleMeasures_filtered.npy", Filtered_Measurements[0:4, :])
    
    

########################################################
#************ VISUALIZE MEASURES           ***********#
#######################################################

if make_plots:
    for input_file_name in input_file_names:
        base_path          = os.path.dirname(input_file_name)+"/"
        orig_volume             = np.fromfile(os.path.dirname(input_file_name)+"/volume.raw", dtype=np.uint8).reshape(volume_shape)
        result_volume           = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
        
        morph_drain = result_volume.copy()
        morph_drain[(result_volume != 1) & (result_volume != 2)]          = 0
        Measurements = np.load(base_path+"AngleMeasures.npy")
        
        angles  = Measurements[0,:]
        coord_x = Measurements[1,:].astype(int)
        coord_y = Measurements[2,:].astype(int)
        coord_z = Measurements[3,:].astype(int)
        
        volume              = morph_drain.copy()
        solid_mask          = (morph_drain != 1) & (morph_drain != 2)
        volume[solid_mask]  = 0
        volume[~solid_mask] = 1
        
        volume_wAngles                            = volume.copy()
        volume_wAngles[coord_x, coord_y, coord_z] = angles
        
    
        
        pl.Plot_Classified_Domain(volume, base_path+"_volume", 
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
                                      1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                                  },
                                  show_label=False,
                                  show_edges=False, lighting=True,
                                  smooth_shading=True, split_sharp_edges=False,
                                  ambient=0.3, diffuse=None, specular=None
                                  )
        
        pl.Plot_Classified_Domain(morph_drain, base_path+"_morph_drain", 
                                  remove_value=[2],
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 1),  # Assign grey for solid cells
                                      1: (0.8, 0.36, 0.36, 1),  # Assign red for fluid I
                                      2: (0.39, 0.58, 0.93, 1),  # Assign blue for fluid II
                                  },
                                  show_label=False,
                                  show_edges=False, lighting=True,
                                  smooth_shading=True, split_sharp_edges=True,
                                  ambient=0.3, diffuse=None, specular=None
                                  )
        
        sub_volume = volume_wAngles[100:200,150:200,0:100]
        pl.Plot_Classified_Domain(sub_volume, base_path+"sub_volume_wAngles", 
                                  remove_value=[1],
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 0.3), # Assign grey for solid cells
                                      1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                                  },
                                  show_label=False,
                                  split_sharp_edges=True
                                  )
        
        pl.Plot_Classified_Domain(volume_wAngles, base_path+"volume_wAngles", 
                                  remove_value=[1],
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 0.1), # Assign grey for solid cells
                                      1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                                  },
                                  show_label=False,
                                  split_sharp_edges=True
                                  )
        
        hollow_volume               = morph_drain.copy()
        solid_mask                  = (morph_drain != 1) & (morph_drain != 2)
        hollow_volume[solid_mask]   = 0
        hollow_volume[~solid_mask]  = 1
        hollow_volume               = util.Remove_Internal_Solid(hollow_volume)
        print("Surface cells:",     np.count_nonzero(hollow_volume==0))
        print("Measured cells: ",   Measurements.shape[1])
        print(Measurements.shape[1])
        print("(Measured /Surface cells) %: ", 100*Measurements.shape[1]/np.count_nonzero(hollow_volume==0))
    
    
    
    
        # ANALYZING FILTERED MEASURMENTS
        base_path          = os.path.dirname(input_file_name)+"/"
        result_volume           = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
        morph_drain = result_volume.copy()
        morph_drain[(result_volume != 1) & (result_volume != 2)]          = 0
        Measurements_filtered = np.load(base_path+"AngleMeasures_filtered.npy")
        
        angles  = Measurements_filtered[0,:]
        coord_x = Measurements_filtered[1,:].astype(int)
        coord_y = Measurements_filtered[2,:].astype(int)
        coord_z = Measurements_filtered[3,:].astype(int)

        volume              = morph_drain.copy()
        solid_mask          = (morph_drain != 1) & (morph_drain != 2)
        volume[solid_mask]  = 0
        volume[~solid_mask] = 1
        
        volume_wAngles                            = volume.copy()
        volume_wAngles[coord_x, coord_y, coord_z] = angles
        
        sub_volume = volume_wAngles[100:200,150:200,0:100]
        pl.Plot_Classified_Domain(sub_volume, "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Slide Images/Exemplo Medicoes/sub_volume_wAngles_filtered", 
                                  remove_value=[1],
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 0.3), # Assign grey for solid cells
                                      1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                                  },
                                  show_label=False,
                                  split_sharp_edges=True
                                  )
        
        pl.Plot_Classified_Domain(volume_wAngles, "/home/gabriel/Desktop/Molhabilidade/ContactAngle_Interpolation-main --- INTERPORE TEST CODE/Slide Images/Exemplo Medicoes/volume_wAngles_filtered", 
                                  remove_value=[1],
                                  special_colors= {
                                      0: (0.5, 0.5, 0.5, 0.1), # Assign grey for solid cells
                                      1: (0.0, 0.0, 0.0, 1)  # Assign black for void cells (removed from plot)
                                  },
                                  show_label=False,
                                  split_sharp_edges=True
                                  )
    
        hollow_volume               = morph_drain.copy()
        solid_mask                  = (morph_drain != 1) & (morph_drain != 2)
        hollow_volume[solid_mask]   = 0
        hollow_volume[~solid_mask]  = 1
        hollow_volume               = util.Remove_Internal_Solid(hollow_volume)
        print("Surface cells:",     np.count_nonzero(hollow_volume==0))
        print("Measured cells: ",   Measurements_filtered.shape[1])
        print(Measurements_filtered.shape[1])
        print("(Measured /Surface cells) %: ", 100*Measurements_filtered.shape[1]/np.count_nonzero(hollow_volume==0))
