import numpy as np
from Utiliy import Create_kinetc_signal , Generate_matrix_withrate

def Check_best_residualts(time, Signal, model_matrix, taus, Spectra):

    # generate parameters

    N_states = 5
    t_0 = 0
    Scores = np.zeros(1)
    GS_spectra = np.zeros([256,1])
    Spectra = np.concatenate((Spectra,GS_spectra), axis=1)

    probe = np.abs(taus)
    probe[1:][probe[1:] < 0.9] = 0
    probe = probe[probe != 0]

    for jj in range (len(probe)):

        probe[jj] = 1/probe[jj]

    try:

        matrix_with_k = Generate_matrix_withrate(N_states, model_matrix , probe)
    
    except:
        
        single_Solution= np.zeros([256,256])
    
    else:
        Kinetic_signal = Create_kinetc_signal(matrix_with_k, time, t_0, time_before_zero = False)
        single_Solution = np.matmul(Spectra,Kinetic_signal.T)

    Scores[0] = np.sum(np.sqrt((Signal - single_Solution)**2))

    return single_Solution, Scores, Kinetic_signal


def Check_best_residualts_GEL(time, Signal, model_matrix, taus, Spectra):

    # generate parameters

    N_states = 5
    t_0 = 0
    Scores = np.zeros(1)
    GS_spectra = np.zeros([256,1])
    Spectra = np.concatenate((Spectra,GS_spectra), axis=1)
    t_steps = np.array([1,2,5,10,15,20,25,30,40,50,60,80,100,150,200,300,400,600])
       
    Gel_img = np.zeros((256,256))
    step_time = int(Gel_img.shape[1]/t_steps.shape[0]);
    pix_const = 3;

    probe = np.abs(taus)
    probe[1:][probe[1:] < 0.9] = 0
    probe = probe[probe != 0]

    for jj in range (len(probe)):

        probe[jj] = 1/probe[jj]

    try:

        matrix_with_k = Generate_matrix_withrate(N_states, model_matrix , probe)
    
    except:
        
        Gel_img = np.zeros([256,256])
    
    else:
        Kinetic_signal = Create_kinetc_signal(matrix_with_k, time, t_0, time_before_zero = False)

    for j in range (N_states - 1):
           
        for k in range(len(t_steps)):
                            
            Gel_band = np.zeros([Gel_img.shape[0], Gel_img.shape[1]])
            specific_time = np.where(time <= t_steps[k])[0]

            Gel_band[step_time*k : step_time*(k+1) - pix_const,:] = Spectra[:,j]
            Gel_img = Gel_img + Gel_band *Kinetic_signal[len(specific_time),j]
            
    Scores[0] = np.sum(np.sqrt((Signal - Gel_img.T)**2))

    return Gel_img.T, Scores, Kinetic_signal