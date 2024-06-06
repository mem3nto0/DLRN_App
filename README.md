# DLRN_App

The Deep Learning Reaction Network (DLRN) App is a user-friendly graphical interface that can be used to analyze
time-resolved spectroscopy and agarose gel data sets. To use DLRN app, please follow the protocol:

1) close the GitHub page using "git clone https://github.com/mem3nto0/DLRN_App.git"
2) unzip the pre-trained model files
3) start the GUI using the command "python3 DLRN_GUI.py" in the prompt or using VScode.

After these three steps, a Graphic window will open and a few options can be selected:

1.	Select Spectra or Agarose Gel from the first set of options. This will load the pre-trained DLRN model for the respective scenario. This can take a few minutes.
2.	 Select Top 1 or Top 3 from the second set of options. This will change the analysis output, giving the solution for either the most probable output or the three most probable outputs.
3.	Select the scale factor (suggested value = 1). This rescales the timescale to let DLRN analyze data sets with a time window larger than one timescale. However, using a large value for the scale factor can change the     
    results of the analysis due to data interpolation during preprocessing data preparation.
4.	Load the timescale to be used for the measurements. It is important to rescale the data with one that matches the timescale used during the DLRN analysis. 
5.	Load the data to be analyzed using “load the data” (located at the bottom). Search for the data that you want to analyze using the browser window. Only a NumPy zip file (.npz) having a subfolder “train” or .txt/.dat 
    files can be loaded in the GUI.
6.	(Optional) Is it possible to test the DLRN performance using the “Test DLRN” button. This allows the user to try a few ground truth data to check the performance. 
7.	Click the “Data Analysis” button to start the analysis and obtain the DLRN analysis results. This can be done only after the compulsory steps (1–5) have been completed.

This is a collaboration partnership with the group of Prof. Dr. Susanne Gerber, Uni Medical Center, Mainz.


