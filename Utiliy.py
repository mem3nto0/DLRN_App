#%%
import numpy as np
from scipy.integrate import odeint

"""
///////////////////////////////////////////
/// new version to generate the k values ////
//////////////////////////////////////////
"""

def Generate_kvalues(Model_pred, N_real_states, time_const_range):

    k_values = [];
    index_chosen = [];
    time = [];
    y_tau_value = np.zeros(len(time_const_range))
    scale_index = np.arange(0,int(len(time_const_range)),1)
    start = np.zeros(1)

    index_chosen.append(int(start))

    if N_real_states == 1:

        index_chosen.append(int(len(time_const_range)))

    else:

        for j in range(N_real_states-1):

            if j == 0:

                rnd_choise = int(np.random.choice(scale_index[20:int(len(scale_index)/2.5)]))
                index_chosen.append(rnd_choise)
                new_range = scale_index[rnd_choise:]

            else:

                rnd_choise = int(np.random.choice(new_range[30:int(len(new_range)/2)]))
                index_chosen.append(rnd_choise)
                new_range = new_range[int(np.where(new_range == rnd_choise)[0]):]

        index_chosen.append(int(len(time_const_range)))

    Count_states = 0;

    for i in range(Model_pred.shape[0]):

        probe = int(np.abs(Model_pred[i,i]))

        if probe > 0:
            Count_states +=1;

        for jj in range (probe):

            if jj== 0:

                probe_time = np.random.choice(time_const_range[index_chosen[Count_states -1]:index_chosen[Count_states]])                               
                k_values.append(1/probe_time)
                time = probe_time
                y_tau_value[np.where(time_const_range == probe_time)[0]] = 1

            else:

                probe_time = np.random.choice(time_const_range[index_chosen[Count_states -1]:index_chosen[Count_states]])
                
                while probe_time < time*0.5 or probe_time > time*2:
                    probe_time = np.random.choice(time_const_range[index_chosen[Count_states -1]:index_chosen[Count_states]])

                k_values.append(1/probe_time)                    
                time = probe_time
                y_tau_value[np.where(time_const_range == probe_time)[0]] = 1

    return k_values , y_tau_value


"""
//////////////////////////////////////////////////
/// Create kinetic traces from a kinetic model ////
//////////////////////////////////////////////////

"""

def Create_kinetc_signal(matrix_with_k, time, t_0, time_before_zero):

    def Resolve_ODE(z, time, matrix_with_k, t_0):

        dzdt = np.zeros(matrix_with_k.shape[0]) 

        irf = np.exp(-2*((time-t_0)/0.1)**2)

        for i in range (matrix_with_k.shape[0]):

            if i == 0:              
                dzdt[i] = np.matmul(matrix_with_k[i,:],z.T) # + irf
            else:
                dzdt[i] = np.matmul(matrix_with_k[i,:],z.T)
        return dzdt

    if time_before_zero == True:

    # initial condiction
        Initial = np.zeros(matrix_with_k.shape[0])


        # resolve differential equation
        z = np.zeros([len(time),matrix_with_k.shape[0]])

        for i in range(1,len(time)): #this for loop is usefull to stablish max(z) = z(t_0)
        
            t_span = [time[i-1],time[i]]
            z_step = odeint(Resolve_ODE, Initial, t_span, args=(matrix_with_k,t_0))

            z[i,:] = z_step[1]
            Initial = z_step[1]

    else:

        Initial = np.zeros(matrix_with_k.shape[0])
        Initial[0] = 1
        
        z = odeint(Resolve_ODE, Initial, time, args=(matrix_with_k,t_0))
        
    return z


"""
//////////////////////////////////////////////////
/// Generate Spectra for each spectral component ////
//////////////////////////////////////////////////

"""

def Spectra_generator(N_real_states):

    wl = np.arange(0, 256, 1)
    N_Gaus = 8;
    wl_signal = np.zeros((len(wl), N_real_states))

    for i in range (N_real_states - 1):
            
        for j in range(np.random.randint(1,N_Gaus)):
            
            if j == 0:
                amp = np.random.randint(4,10)
            
            else:
                amp = np.random.randint(2,10)

            sigma = np.random.randint(3,26)
            w0 = 2*np.random.randint(1, 128)
            wl_signal[:,i] = wl_signal[:,i] + amp*np.exp(-0.5*((wl-w0)/sigma)**2)                
        
        wl_signal[:,i] = wl_signal[:,i]

    return wl_signal / np.max(wl_signal)


"""
////////////////////////////////////////////////////////
/// associate to each pathway the kinetic rate constant, ////
/// generating the kinetic matrix with rate constants    ////
/////////////////////////////////////////////////////////

"""

def Generate_matrix_withrate(N_species, Model_pred, k_values):

    matrix_with_k = np.zeros([N_species,N_species])

    k_use = k_values.copy()

    for i in range (N_species):

        for j in range (N_species):
            
            probe = Model_pred[j,i]

            if probe <= -1:

                for n in range(int(abs(probe))):

                    matrix_with_k[j,i]  = matrix_with_k[j,i]  - k_use[n]

            if probe >= 1:

                for n in range(int(abs(probe))):

                    matrix_with_k[j,i]  = matrix_with_k[j,i]  + k_use[n]
                    k_use = np.delete(k_use, 0) 

    return matrix_with_k
