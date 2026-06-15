# Data-Driven-Thermal-Modeling-Wire-Arc-Additive-Manufacturing-Process
Data-driven thermal modeling of WAAM using Neural Network Approach
__________________________________________________________________________________________________________________________________________
**Overview**
This file contains neural-network surrogate modelling work developed for a master-thesis on data-driven thermal modeling of Wire Arc Additive Manufacturing process.  
The input data for NN model training is obtained from *Simufact Welding 2022 software* where author performed 25 Finite Element Simulations by creating 25 Design of Experiments. The data coming from each simulation is stored in the Parquet file format. The developed NN model is trained 25 times (for all 25 DOE cases). 
__________________________________________________________________________________________________________________________________________
**Project Workflow**
1. Load the pre_processed files in the parquet file format (Simulation data).
2. Train the Neural Network model for each DOE case.
3. Save the 25 trained models in .h5 file format.
4. Evaluate each model using Root Mean Squared Error (RMSE) , Mean Absolute Error (MAE), and coefficient of determination (R^2).
5. Generate training, validation losses plots , scatter plots, NN_predicted-verses-actual temperature-time plots for all 25 Experiments.
________________________________________________________________________________________________________________________________________
Now we will go through project structure. NN_WAAM_ALL is a main big folder containing everthing related to the NN_WAAM, such as data files, main python code for Neural Network model training, trained models, evaluation plots, training plots, evaluation results, python file for NN_predicted and original temperature vs time plots (for all 25 DOE), temperature-verses-time plots for test set for all 25 DOE. 
NN_WAAM_ALL/
│
├── data/
│   └── pre_processed/
│
├── evaluation_plots/
├── evaluation_plots_temp_time_test_perDOE/
├── evaluation_results/
├── trained_models/
├── training_plots/
│
├── surrogate_NN_model_training_25_DOE_maincode.py
├── plot_temp_vs_time_per_doe_test_NN.py
└── README.md
________________________________________





