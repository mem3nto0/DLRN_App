import numpy as np
import matplotlib.pyplot as plt
import os

"""every data is saved always for A,B,C,B and the time constants for tau1,tau2,tau3..."""

def Save_analysis(ind,path, pre_amp, pre_tau, Kinetic_signal):

    inner_folder = os.path.join(path, f'Solution_{ind}')
    os.makedirs(inner_folder, exist_ok=True) 

    amp_path = os.path.join(inner_folder, 'Amplitude_solution.txt')
    np.savetxt(amp_path, pre_amp)

    tau_path = os.path.join(inner_folder, 'Tau_solution.txt')
    np.savetxt(tau_path, pre_tau)

    kinetic_path = os.path.join(inner_folder, 'Kinetic_solution.txt')
    np.savetxt(kinetic_path, Kinetic_signal)

    """
    binary_model_path = os.path.join(inner_folder, 'Binary_model_solution.txt')
    np.savetxt(binary_model_path, pre_model)
    """
    
    plot_file_path = os.path.join(inner_folder, 'Final_analysis_graph.png')
    plt.savefig(plot_file_path)