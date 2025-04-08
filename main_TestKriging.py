import utilities as util
import numpy as np
import Plotter as pl
import matplotlib.pyplot as plt



fluid_default_value=1
distance_function = util.euclidean
input_file_name = "/home/gabriel/Desktop/Molhabilidade/WettabilityInterpolation/Rock Volumes/Iniciais/Example_11.raw"
volume_shape = (50,50,50)#(100,25,25)

sampled_volume = np.fromfile(input_file_name, dtype=np.uint8).reshape(volume_shape)
pl.Plot_Classified_Domain(sampled_volume, filename="sampled_volume")
pl.Plot_Domain(sampled_volume, filename="sampled_volume2", remove_value=[1], colormap='hot')


hollow = util.Remove_Internal_Solid(sampled_volume, keep_boundary=True)
pl.Plot_Classified_Domain(hollow, filename="hollow", remove_value=[1])

volume = (~(sampled_volume != fluid_default_value)).astype(int)
pl.Plot_Classified_Domain(volume, filename="volume")




samples_coord = util.COLLECT_SAMPLES_COORDINATES(sampled_volume)
samples_values = sampled_volume[tuple(np.array(samples_coord).T)]


# Validar outras formas de variograma
# Validar busca por vizinhos
num_workers = 3
N_neighbors = None

model = 'gaussian'

# My Kriging implementation
interpolated_volume = util.KRIGING_INTERPOLATION(
    sampled_volume, 
    samples_coord, 
    distance_function, 
    N_neighbors=N_neighbors, 
    solid_default_value=0, 
    num_workers=num_workers, 
    model=model,
    universal=True,
    keep_boundary=True,
    variogram_sampling=1)
pl.Plot_Domain(interpolated_volume, filename="interpolated_volume", remove_value=[1], colormap='hot')
plt.hist(interpolated_volume[(interpolated_volume!=0)&(interpolated_volume!=1)])
plt.title("MY")
plt.show()




"""
# Pykrige implementation
from pykrige.uk3d import UniversalKriging3D
import numpy as np
x, y, z = np.array(samples_coord).T
values = samples_values.astype(float)  # Make sure values are float
UK3D = UniversalKriging3D(
    x, y, z, values,
    variogram_model=model,
    nlags=n_lags,
    enable_plotting=True
)
domain_solid_coord = np.argwhere(hollow == 0)
X = domain_solid_coord[:,0].astype(np.float64)
Y = domain_solid_coord[:,1].astype(np.float64)
Z = domain_solid_coord[:,2].astype(np.float64)
solid_pykrige, variance = UK3D.execute('points', X.flatten(), Y.flatten(), Z.flatten(), backend='loop')
interpolated_pykrige = hollow.copy()
interpolated_pykrige[hollow == 0] = solid_pykrige
plt.hist(interpolated_pykrige[(interpolated_pykrige!=0)&(interpolated_pykrige!=1)])
plt.title("PYKRIGE")
plt.show()
pl.Plot_Domain(interpolated_pykrige, filename="interpolated_pykrige", remove_value=[1], colormap='hot')
"""