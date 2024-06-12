#%%
import sys
import numpy as np
import Global_variable
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from print_table import Table_print
from skimage.transform import resize
from draw_model import model_drawing
from plot_results import plotting_results, plotting_results_test
from Save_data import Save_analysis
from Analyse_residuals import Check_best_residualts, Check_best_residualts_GEL
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt  # Import Qt module from PyQt5
import PyQt5

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        gpus = tf.config.experimental.list_physical_devices('GPU')

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

        self.setStyleSheet("background-color: #444444; color: #ffffff;")

        self.setWindowTitle("DLRN APP")
        self.resize(400, 300)  # Width, Height

        # Create the main widget and layout
        main_widget = QWidget(self)
        layout = QVBoxLayout(main_widget)

        # Create the second QComboBox and add it to the layout
        self.combo_box_1 = QComboBox(self)
        self.combo_box_1.addItems(["---","Spectra", "Agarose Gel"])
        layout.addWidget(self.combo_box_1)

        # Create the second QComboBox and add it to the layout
        self.combo_box_2 = QComboBox(self)
        self.combo_box_2.addItems(["---","Top1", "Top3"])
        layout.addWidget(self.combo_box_2)

        # Create the label "Adjusting Factor"
        self.adjusting_factor_label = QLabel("Adjusting Factor (recommended:1 max_val:10):")
        layout.addWidget(self.adjusting_factor_label, alignment=Qt.AlignLeft)

        # Create the QLineEdit widget
        self.adjusting_factor_textbox = QLineEdit()
        self.adjusting_factor_textbox.setText("-")
        self.adjusting_factor_textbox.setFixedWidth(100)
        self.adjusting_factor_textbox.textChanged.connect(self.update_id)
        layout.addWidget(self.adjusting_factor_textbox)

        self.load_Files = QPushButton("Load custom time scale", self)
        self.load_Files.clicked.connect(self.load_time_scale)
        layout.addWidget(self.load_Files)

        self.load_Files = QPushButton("Load data", self)
        self.load_Files.clicked.connect(self.load_data)
        layout.addWidget(self.load_Files)

        self.load_Files = QPushButton("test DLRN", self)
        self.load_Files.clicked.connect(self.test_DLRN)
        layout.addWidget(self.load_Files)

        self.load_Files = QPushButton("Data analysis", self)
        self.load_Files.clicked.connect(self.Analyse_data)
        layout.addWidget(self.load_Files)

        # Add stretch to push the label to the left
        layout.addStretch()

        # Set the main widget and layout
        self.setCentralWidget(main_widget)

        # Connect signals (optional)
        self.combo_box_1.currentIndexChanged.connect(self.on_combobox1_changed)
        self.combo_box_2.currentIndexChanged.connect(self.on_combobox2_changed)

    def update_id(self, text):
        try:

            Global_variable.adjusting_factor = float(text)

        except ValueError:
            None
            # Handle non-numeric input gracefully
            pass

    def on_combobox1_changed(self, index):

        selected_option = self.combo_box_1.itemText(index)

        def top_3_acc(y_true, y_pred):
            return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
        
        if selected_option == "Spectra":
            path = os.getcwd()
            path_model = path + "/pretrained_model_Spectra/resV2_spectra_model"
            model_model = tf.keras.models.load_model(path_model, custom_objects={'top_3_acc':top_3_acc})
            Global_variable.model_model = model_model

            path_amp = path + "/pretrained_model_Spectra/resV2_spectra_amplitude"
            model_amp = tf.keras.models.load_model(path_amp)
            Global_variable.model_amp = model_amp

            path_tau = path + "/pretrained_model_Spectra/resV2_spectra_Tau"
            model_tau = tf.keras.models.load_model(path_tau)
            Global_variable.model_tau = model_tau

        if selected_option == "Agarose Gel": # to change for the GEL path
            path = os.getcwd()
            path_model = path + "/pretrained_model_Gel/resV2_GEL_takemodel"
            model_model = tf.keras.models.load_model(path_model, custom_objects={'top_3_acc':top_3_acc})
            Global_variable.model_model = model_model

            path_amp = path + "/pretrained_model_Gel/resV2_GEL_takespectra+dilat"
            model_amp = tf.keras.models.load_model(path_amp)
            Global_variable.model_amp = model_amp

            path_tau = path + "/pretrained_model_Gel/resV2_GEL_Tau(MSE)"
            model_tau = tf.keras.models.load_model(path_tau)
            Global_variable.model_tau = model_tau

        Global_variable.Kind_analysis = selected_option

    def on_combobox2_changed(self, index):

        selected_option = self.combo_box_2.itemText(index)

        if selected_option == "Top1":

            Global_variable.kind_Top = 1

        if selected_option == "Top3":

            Global_variable.kind_Top = 3        

    def load_time_scale(self):

        adjusting_factor = Global_variable.adjusting_factor

        file_path, _ = QFileDialog.getOpenFileName()
        time_original = np.loadtxt(file_path)

        if adjusting_factor == -1:

            adjusting_factor = 1000/np.max(time_original)

        else:
            adjusting_factor = 1/adjusting_factor

        Global_variable.adjusting_factor = adjusting_factor
        Global_variable.time_original = time_original*adjusting_factor

    def load_data(self):

        file_path, _ = QFileDialog.getOpenFileName()

        try:

            data_general = np.load(file_path)
            data = data_general[data_general.files[0]]

        except:
            
            data = np.loadtxt(file_path)

        time_data = Global_variable.time_original # i need few passage to make the time for training and real data the same
        time_training = Global_variable.time # right now, time data and time training are the same. to check how to fix it
              
        norm_value = np.max(data)
        data = data / norm_value        
        
        if data.shape[0] < 256:

            data_resized = resize(data, (256,data.shape[1]), anti_aliasing= True)
        
        else:

            data_resized = data
                
        Final_data = np.zeros([256,256])

        for j in range(256):

            Final_data[j,:] = np.interp(time_training, time_data, data_resized[j,:])

        Global_variable.data_to_analyse = Final_data
        Global_variable.norm_value = norm_value

        wl = Global_variable.wl

        selected_option = Global_variable.Kind_analysis
        
        
        if selected_option == "Spectra":
            
            plt.figure(1)   
            plt.contourf(time_training,wl,Final_data)
            plt.show()

        else:

            plt.figure(1)   
            plt.contourf(Final_data)
            plt.show()

    def test_DLRN(self):

        selected_option = Global_variable.Kind_analysis

        if selected_option == "Spectra":

            path = os.getcwd()
            path = path + "/Test_data_APP/Spectra"

        else: 

            path = os.getcwd()
            path = path + "/Test_data_APP/Gel"

        list = os.listdir(path)
        rand_data = np.random.choice(list)

        with np.load(path + "/" + rand_data) as data_loaded: #"/" + rand_data

            data = data_loaded["train"]
            #data = data/np.max(data)
            data = np.expand_dims(data,axis = -1)
            data = np.expand_dims(data,axis = 0)

            true_tau = data_loaded["label1"]
            true_model = data_loaded["label2"]
            true_amp = data_loaded["label3"]

        path = os.getcwd()
        dirs = path + "/Model_information"

        with np.load(dirs + "/onehot-reppresentation + binary-matrix.npz") as data_info:

            Binary_matrix = data_info["binary_matrix"]

        model_model = Global_variable.model_model
        model_tau = Global_variable.model_tau
        model_amp = Global_variable.model_amp
        Time = Global_variable.time
        kind_top = Global_variable.kind_Top

        pre_model = model_model.predict(data)
        top_3_indices = np.argpartition(pre_model[0,:],-int(kind_top))[-int(kind_top):]

        plt.figure(1)
        plt.title("model prediction one-hot encoding")
        plt.plot(pre_model[0,:],marker = "o", color = "red", label = "prediction")
        plt.plot(true_model, marker = "o", color = "black", label = "expectation")
        plt.ylabel('probability confidence', fontsize=12)
        plt.xlabel('model index', fontsize=12)
        plt.legend()
        plt.show()

        Tau_solutions = np.zeros([len(top_3_indices), 7])

        for i in range (len(top_3_indices)):

            model_binary = Binary_matrix[top_3_indices[i]]
            model_binary = np.expand_dims(model_binary, axis = 0)
            X_total = {"input_1": data,"input_2": model_binary}

            pre_tau = model_tau.predict(X_total)
            pre_amp = model_amp.predict(X_total)

            pre_tau[0,1:][pre_tau[0,1:] < 0.8] = 0
            Tau_solutions[i,:] = pre_tau[0,:]

            if selected_option == "Spectra":

                Fit, score, Kinetic_signal = Check_best_residualts(Time, data[0,:,:,0], model_binary[0,:,:], pre_tau[0,:], pre_amp[0,:,:])
            
            else:

                Fit, score, Kinetic_signal = Check_best_residualts_GEL(Time, data[0,:,:,0], model_binary[0,:,:], pre_tau[0,:], pre_amp[0,:,:])

            plotting_results_test(model_binary, 
                            pre_model, 
                            i, 
                            top_3_indices, 
                            pre_amp, 
                            Time, 
                            Kinetic_signal, data, Fit, score,
                            true_amp)

        Table_print(Tau_solutions, true_tau, kind_top)


    def Analyse_data(self):

        data = Global_variable.data_to_analyse
        data = np.expand_dims(data, axis= -1)
        data = np.expand_dims(data, axis= 0)

        path = os.getcwd()
        dirs = path + "/Model_information"

        with np.load(dirs + "/onehot-reppresentation + binary-matrix.npz") as data_info:

            Binary_matrix = data_info["binary_matrix"]

        model_model = Global_variable.model_model
        model_tau = Global_variable.model_tau
        model_amp = Global_variable.model_amp
        Time = Global_variable.time
        wl = Global_variable.wl
        selected_option = Global_variable.Kind_analysis
        kind_top = Global_variable.kind_Top
        adjusting_factor = Global_variable.adjusting_factor

        pre_model = model_model.predict(data)
        top_3_indices = np.argpartition(pre_model[0,:],-int(kind_top))[-int(kind_top):]

        plt.figure(1)
        plt.title("model prediction one-hot encoding")
        plt.plot(pre_model[0,:],marker = "o", color = "red", label = "prediction")
        plt.ylabel('probability confidence', fontsize=12)
        plt.xlabel('model index', fontsize=12)
        plt.legend()
        plt.show()

        Tau_solutions = np.zeros([len(top_3_indices), 7])

        # Create a folder to save the analysis
        folder_name = 'Analysis_solutions'
        os.makedirs(folder_name, exist_ok=True)

        for i in range (len(top_3_indices)):

            model_binary = Binary_matrix[top_3_indices[i]]

            N_taus = 0

            for kk in range (5):

                N_taus = N_taus + np.abs(model_binary[kk,kk])

            N_taus =int(N_taus)

            model_binary = np.expand_dims(model_binary, axis = 0)
            X_total = {"input_1": data,"input_2": model_binary}

            pre_tau = model_tau.predict(X_total)
            pre_amp = model_amp.predict(X_total)

            pre_tau[0,N_taus:] = 0

            Tau_solutions[i,:] = pre_tau[0,:]/adjusting_factor

            if selected_option == "Spectra":

                Fit, score, Kinetic_signal = Check_best_residualts(Time, data[0,:,:,0], model_binary[0,:,:], pre_tau[0,:], pre_amp[0,:,:])
            
            else:

                Fit, score, Kinetic_signal = Check_best_residualts_GEL(Time, data[0,:,:,0], model_binary[0,:,:], pre_tau[0,:], pre_amp[0,:,:])

            plotting_results(model_binary, 
                            pre_model, 
                            i, 
                            top_3_indices, 
                            pre_amp, 
                            Time, 
                            Kinetic_signal, data, Fit, score)

            Inner_folder = path +"/"+ folder_name
            Save_analysis(i, Inner_folder, pre_amp[0,:,:], Tau_solutions[i,:], Kinetic_signal)

        true_tau = np.zeros(7)
        Table_print(Tau_solutions, true_tau, kind_top)
 

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
