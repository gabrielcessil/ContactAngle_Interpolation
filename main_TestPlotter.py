import pyvista as pv
import numpy as np

def basic_test():

    # Create the 3D NumPy array of spatially referenced data
    values = np.linspace(0, 10, 1000).reshape((20, 5, 10))

    # Create the spatial reference
    grid = pv.ImageData()

    # Set the grid dimensions: shape because we want to inject our values on the POINT data
    grid.dimensions = values.shape

    # Edit the spatial reference
    grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.point_data["values"] = values.flatten(order="F")  # Flatten the array

    # Save the plot as an image
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True to avoid GUI issues
    plotter.add_mesh(grid, show_edges=True)
    plotter.screenshot("grid_plot.png")  # Save as PNG
    plotter.close()
    

basic_test()

import Plotter as pl

def test_plot_domain():
    # Create a 3D numpy array with sample values
    values = np.random.randint(0, 5, size=(10, 10, 10))  # Example 3D domain
    filename = "test_plot_domain"
    remove_value = [3]  # Example value to remove
    special_labels = {2: "red", 4: "blue"}  # Example special labels
    
    pl.Plot_Domain(values, filename, remove_value=remove_value, special_labels=special_labels)
    print("Plot_Domain executed successfully.")

def test_plot_classified_domain():
    # Create a 3D numpy array with labeled classes
    values = np.random.randint(0, 5, size=(10, 10, 10))
    filename = "test_plot_classified_domain"
    remove_value = [3]  # Remove specific class
    labels = {0: "Medium", 1: "Solid", 2: "Fluid", 4: "Gas"}  # Example labels
    
    pl.Plot_Classified_Domain(values, filename, remove_value=remove_value, labels=labels)
    print("Plot_Classified_Domain executed successfully.")

    
test_plot_domain()
test_plot_classified_domain()
