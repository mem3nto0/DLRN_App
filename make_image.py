#%%

#% load the original files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

file_path_main = "Data_collaborator_202404/VisualDSD_simulations_4_extended_ASCI/VisualDSD_simulations_4_extended_ASCI"
path_folder = "/5-ver3_A-B-C-D toehold varied"

file_path_1 = file_path_main + path_folder + "/In-10-Su1-10-Su2-10-Re-30_c.dat"

#file_path_main = "data from collaborator_202401/Nicolo_numerical_data_t1_varied/Nicolo_numerical_data_t1_varied"
#file_path_1 =file_path_main + "/t1c.csv"

# Replace 'your_file.dat' with the path to your .dat file
# You may need to specify additional parameters depending on the structure of your file
data = pd.read_csv(file_path_1, delimiter='\t')

#data = pd.read_csv(file_path_1, delimiter=',')

time_scale_pd = data.iloc[1:,0].values
time_scale_pd = time_scale_pd.astype(float)[:]

trace_A = data.iloc[1:,1].values.astype(float)[:]
trace_B = data.iloc[1:,2].values.astype(float)[:]
trace_C = data.iloc[1:,3].values.astype(float)[:]
trace_D = data.iloc[1:,4].values.astype(float)[:]

file_path_spc = "Data_collaborator_202404/Re__VisualDSD_simulations_--_extended"
path_spectra_B = file_path_spc + "/StateB_Cy5.5.dat"
path_spectra_C = file_path_spc + "/StateC_Atto425.dat"
path_spectra_D = file_path_spc + "/StateD_Cy3.dat"

# Replace 'your_file.dat' with the path to your .dat file
# You may need to specify additional parameters depending on the structure of your file
data_spc_B = pd.read_csv(path_spectra_B, delimiter='\t')
wl_B = data_spc_B.iloc[1:,0].values.astype(float)
spectra_B = data_spc_B.iloc[1:,1].values.astype(float)

data_spc_C = pd.read_csv(path_spectra_C, delimiter='\t')
wl_C = data_spc_C.iloc[1:,0].values.astype(float)
spectra_C = data_spc_C.iloc[1:,1].values.astype(float)

data_spc_D = pd.read_csv(path_spectra_D, delimiter='\t')
wl_D = data_spc_D.iloc[1:,0].values.astype(float)
spectra_D = data_spc_D.iloc[1:,1].values.astype(float)

plt.plot(time_scale_pd,trace_A, label ="A")
plt.plot(time_scale_pd,trace_B, label ="B")
plt.plot(time_scale_pd,trace_C, label ="C")
plt.plot(time_scale_pd,trace_D, label ="D")
plt.xlabel("time")
plt.ylabel("conc.")
plt.legend()
plt.xlim(0,10000)

#%%
import numpy as np
import matplotlib.pyplot as plt

# ///////// COMPARE FITTING VS DRLN ////

time_1 = np.arange(0,20,0.5)
time_2 = np.logspace(1, 2.3, num=(256-len(time_1)), base=20)
time =  np.concatenate((time_1, time_2), axis = 0)

path = "analysis_DLRN_ver2/Analysis_toehold_c/Solution_2"

#path = "Analysis_solutions/Solution_2"

tau_DLRN = path + "/Tau_solution.txt"
ammp_DLRN = path + "/Amplitude_solution.txt"
kineit_trace = path + "/Kinetic_solution.txt"

tau_DLRN = np.loadtxt(tau_DLRN)
ammp_DLRN = np.loadtxt(ammp_DLRN)
kineit_trace = np.loadtxt(kineit_trace)

path2 = "Analysis_fitting"

ammp_Fit = path2 + "/A-C-D_dynamics_In-10-Su1-20-Su2-0-Re-30_amplitude_V2.txt"
kinetic_Fit = path2 + "/A-C-D_dynamics_In-10-Su1-20-Su2-0-Re-30_kinetic_trace_V2.txt"
ammp_expected = path2 + "/expected_amplitude.txt"
time_scale_data = path2 + "/A-C-D_dynamics_In-10-Su1-20-Su2-0-Re-30_timescale.txt"

ammp_Fit = np.loadtxt(ammp_Fit)
kinetic_Fit = np.loadtxt(kinetic_Fit)
ammp_expected = np.loadtxt(ammp_expected)
time_scale_data = np.loadtxt(time_scale_data)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2, 3))

ammp_expected = ammp_expected / np.max(ammp_expected)
wl_scale =np.arange(0,256,1)

ax1.plot(wl_scale,ammp_DLRN [:,1], linewidth = 1, color = "Green")
ax1.plot(wl_scale[::4],ammp_expected [::4,0] , "o", markersize = 2, linewidth = 2, color = "Green")
ax1.plot(wl_scale,ammp_DLRN [:,2], linewidth = 1, color = "Blue")
ax1.plot(wl_scale[::4], ammp_expected [::4,1] , "o", markersize = 2, linewidth = 2, color = "Blue")
ax1.plot(wl_scale,ammp_DLRN [:,3], linewidth = 1, color = "Red")
ax1.plot(wl_scale[::4],ammp_expected [::4,2] , "o", markersize = 2, linewidth = 2, color = "Red")
ax1.set_xlim(0,256)
ax1.set_ylim(-0.1,1.1)
ax1.set_xlabel("Wl(unit)")
ax1.set_ylabel("norm. I")
ax1.set_xticks([0,100,200])


# Create the custom index array
indices1 = np.arange(0, 30, 6)  # From 0 to 50, every 5 elements
indices2 = np.arange(30, len(time_scale_data), 20)  # From 50 to the end, every 30 elements
custom_indices = np.concatenate([indices1, indices2])


# for the plot you need the coefficient scale factor
ax2.plot(time_scale_data[custom_indices], trace_B[custom_indices] /10 , "o", markersize = 2, color= "Green")
ax2.plot(time*8, kineit_trace[:,1], linewidth = 1, color= "Green")
ax2.plot(time_scale_data[custom_indices], trace_C[custom_indices] /10 , "o", markersize = 2, color= "Blue")
ax2.plot(time*8, kineit_trace[:,2], linewidth = 1, color= "Blue")
ax2.plot(time_scale_data[custom_indices], trace_D[custom_indices] /10 , "o", markersize = 2, color= "Red")
ax2.plot(time*8, kineit_trace[:,3], linewidth = 1, color= "Red")
ax2.set_xlim(0,5000)
ax2.set_ylim(0,1.1)
ax2.set_xlabel("time (sec)")
ax2.set_ylabel("norm. conc.")
"""

ax2.plot(time_scale_data[custom_indices], trace_B[custom_indices] /10 , "o", markersize = 2, color= "Green")
ax2.plot(time_scale_data, kin_fun[:,1], linewidth = 1, color= "Green")
#ax2.plot(time_scale_data[custom_indices], trace_C[custom_indices] /10 , "o", markersize = 2, color= "Green")
#ax2.plot(time_scale_data, kin_fun[:,2], linewidth = 1, color= "Green")
ax2.plot(time_scale_data[custom_indices], trace_D[custom_indices] /10 , "o", markersize = 2, color= "Red")
ax2.plot(time_scale_data, kin_fun[:,2], linewidth = 1, color= "Red")

ax2.set_xlim(0,5000)
ax2.set_ylim(0,1.1)
ax2.set_xlabel("time (sec)")
ax2.set_ylabel("norm. conc.")
"""
plt.tight_layout()

plt.savefig("A-C-B-D_toehold_c_V2.png", dpi = 600, bbox_inches = "tight")

#%%
from Utiliy import Create_kinetc_signal


matrix_with_k = np.array([[-1/150, 0 , 0, 0 ],
                          [1/150, -1/939, 0, 0],
                          [0, 1/939, 0, 0 ],
                          [0, 0, 0 ,0 ]])

kin_fun = Create_kinetc_signal(matrix_with_k, time_scale_data, 0, time_before_zero=False)


#%%

# /// try to fit tau-2 for extrapolate the real time constat tau-2 for A-C-D///


# input 10 
conc_values = np.array([15, 20, 25, 30])
k_values = np.array([1/3292, 1/2400, 1/2020 ,1/1798]) # from DLRN

errors = 0.1*k_values  # Example errors for each data point

# Compute weights from errors
weights = 1 / errors

# Performing linear fit
coefficients = np.polyfit(conc_values, k_values, 1, w = weights)  # The third argument specifies the degree of the polynomial, which is 1 for linear fit
slope, intercept = coefficients

# Printing the slope and intercept
print("Slope:", slope)
print("Intercept:", intercept)

plt.figure(figsize=(3, 2))
# Plotting the data points
plt.errorbar(conc_values, k_values, yerr=errors, fmt='o', label='Data with error bars', color='blue', capsize=5, markersize = 4.5)

scale = np.arange(0,41, 1)

# Plotting the linear fit
plt.plot(scale, slope * scale + intercept, color='red')

# Adding labels and legend
plt.xlabel('Re concentration')
plt.ylabel('k ($M^{-1}$$s^{-1}$)')
plt.xticks(np.array([0,10,20,30,40]))
plt.yticks(np.array([0.00025,0.00050,0.00075,0.001]))
plt.xlim(0,40)
plt.ylim(0,0.00075)
plt.tight_layout()

plt.savefig("Cocentration_Re_vs_k.png",dpi = 600)

#%%
# ///change toehold analysis///
"""
# input 10      toehold:a, b, c, d, e
conc_values = np.array([1, 2, 3, 4, 5])
k_values_AB = np.array([1/80, 1/223, 1/318, 1/387 ,1/281]) # from DLRN
k_values_AC = np.array([1/56, 1/126, 1/213, 1/327 ,1/489]) # from DLRN
"""

# input 10      toehold:a, b, c, d, e
conc_values = np.array([1, 2, 3, 4])
k_values_AB = np.array([1/223, 1/318, 1/387 ,1/281]) # from DLRN
k_values_AC = np.array([1/126, 1/213, 1/327 ,1/489]) # from DLRN

plt.figure(figsize=(3, 2))
# Plotting the data points

plt.plot(conc_values, k_values_AB, "o", label = "k$_A$$_B$")
plt.plot(conc_values, k_values_AC, "o", label = "k$_A$$_C$")
plt.legend()

scale = np.arange(0,41, 1)
ticks = ["a","b","c","d"]
# Adding labels and legend
plt.xlabel('toehold (state C)')
plt.ylabel('k ($M^{-1}$$s^{-1}$)')
plt.xticks(conc_values, ticks)
plt.yticks(np.array([0.004,0.008,0.001]))
plt.ylim(0.0005,0.01)

plt.tight_layout()

"""
plt.xticks(np.array([0,10,20,30,40]))
plt.yticks(np.array([0.00025,0.00050,0.00075,0.001]))
plt.xlim(0,40)
plt.ylim(0,0.00075)

"""

plt.savefig("k_rate_diffirent_toehold_V2.png",dpi = 600)

