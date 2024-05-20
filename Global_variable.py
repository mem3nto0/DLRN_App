
import numpy as np

model_model =[];
model_tau =[];
model_amp = [];
data_to_analyse = [];
norm_value = [];
time_training = [];
Kind_analysis = [];
kind_Top = [];
adjusting_factor = [];
model_test = [];

#generating the time scale for the data analysis

time_1 = np.arange(0,20,0.5)
time_2 = np.logspace(1, 2.3, num=(256-len(time_1)), base=20)
time =  np.concatenate((time_1, time_2), axis = 0)

"""
time1 = np.arange(0,2,0.2)
time2 = np.arange(2,20,0.5)
time3= np.arange(20,50, 1)
time4= np.arange(50,200, 2)
time5 = np.arange(200,460,5)
time6 = np.arange(460,990,10)

time_1 = np.arange(0,2,0.2)       
time_2 = np.arange(2,20,0.5)
time_3 = np.logspace(1, 2.3, num=(256-(len(time_1) + len(time_2))), base=20)
time =  np.concatenate((time_1, time_2,time_3), axis = 0)

time= np.concatenate((time1,time2,time3,time4,time5,time6),axis = 0)
"""
time_original = time;


wl = np.arange(0,256,1)
