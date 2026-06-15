# Data-Driven-Thermal-Modeling-Wire-Arc-Additive-Manufacturing-Process
Data-driven thermal modeling of WAAM using Neural Network Approach
________________________________________________________________________________________________________________________________________
**Overview**

This file contains neural-network surrogate modelling work developed for a master-thesis on data-driven thermal modeling of Wire Arc Additive Manufacturing process.  
The input data for NN model training is obtained from *Simufact Welding 2022 software* where author performed 25 Finite Element Simulations by creating 25 Design of Experiments. The data coming from each simulation is stored in the Parquet file format. The developed NN model is trained 25 times (for all 25 DOE cases). 
________________________________________________________________________________________________________________________________________
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



├── surrogate_NN_model_training_25_DOE_maincode.py

├── plot_temp_vs_time_per_doe_test_NN.py

└── README.md
________________________________________________________________________________________________________________________________________
**Design of Experiments**

A Design of Experiments approach was employed to generate the simulation dataset in this study. In
total, 25 experiments were designed using combinations of five different robot travel speeds and five
distinct wire feed rates. The robot travel speed was varied from 200 to 1000mm/min in increments of
200mm/min (200,400,600,800,1000), while the wire feed rate was varied from 2 to 10m/min in
increments of 2m/min (2,4,6,8,10). Each experiment was identified using the notation robot travel
speed_wire feed rate. For example, in experiment designated with 600_8, 600 represents
the robot travel speed in mm/min, and 8 represents the wire feed rate in m/min.

________________________________________________________________________________________________________________________________________
**Data**

data/pre_processed
This folder contains the preprocessed datasets for all 25 DOE simulations.
Each DOE is stored as a separate Parquet file, for example: 

200_2.parquet

200_4.parquet

200_6.parquet

400_2.parquet

600_4.parquet

800_8.parquet

1000_6.parquet

These (25) files contain the simulation data used for neural-network model training, validation, and testing.

Each parquet file contains column wise information  for each experiment incluidng time, nodal coordinates (x,y,z), and temperature at each node in Kelvin. 

structure of each parquet file:  Node_ID, time, x, y, z, Temp_K

**Main Training Script-Python code**

surrogate_NN_model_training_25_DOE_maincode.py

This is the main python script developed for Nueral Network model training.

## Important Steps Performed by the Script

* Loads each DOE dataset from `data/pre_processed/` in Parquet format using Polars.

* Verifies that every dataset contains the required columns:

  * `time`
  * `x`
  * `y`
  * `z`
  * `Temp_K`

* Uses `time`, `x`, `y`, and `z` as the neural-network input features (covariates) and `Temp_K` as the target variable (dependent variable)

* Splitting of each DOE dataset into:

  * 72% training data
  * 18% validation data
  * 10% test data

* Standardizes the input features and target temperature using `StandardScaler`. The scalers are fitted only on the training data.

* Creation of a feed-forward neural network architecture with:

  * Four input features (time,x,y,z)
  * Dense layers containing 256, 256, 128, 128, 64, and 64 neurons
  * ReLU activation functions
  * A dropout layer with a rate of 0.2 after the first dense layer
  * One output neuron for temperature prediction (T).

* Model is compiled using the Adam optimizer and Mean Squared Error loss.

* Trains a separate neural-network model for every selected DOE case using:

  * Maximum of 50 epochs
  * Batch size of 2048
  * Learning rate of 0.0001
  * Early stopping
  * Automatic learning-rate reduction

* Saves each trained model in `.h5` format inside the `trained_models/` folder.

* Generates training and validation loss plots and saves them in the `training_plots/` folder.

* Evaluates every model on the training, validation, and test datasets using:

  * RMSE
  * MAE
  * R² score
  * Pearson correlation coefficient

* Saves the evaluation metrics (RMSE,MAE,R^2,Pearson Correlation) as CSV files inside the `evaluation_results/` folder. Here, we have 3 CSV files for training, validation, and test datasets. But the most significant file is model_evaluation_metrics_test.csv which stores metrics values for all 25 DOE.

* Creates predicted-versus-actual temperature scatter plots for the training, validation, and test datasets. But plots for test datasets are important. 

* Saves the predicted-versus-actual plots inside the `evaluation_plots/` folder.

___________________________________________________________________________________________________________________________

**Trained Models**

trained_models 

This folder contains the 25 trained neural-network models saved in .h5 format.

For instance:

model_200_2.h5

model_400_4.h5

model_600_6.h5

model_800_8.h5

model_1000_10.h5

Each model corresponds to one DOE Parquet file.
____________________________________________________________________________________________________________________________

**Training Plots**

training_plots

The training_plots folder have training and validation loss curves in .png format for all 25 DOE models.
For example:

training_history_200_2.png

training_history_400_4.png

These curves are used to check convergence of models.
____________________________________________________________________________________________________________________________

**Predicted-Versus-Actual Plots (Scatter plots)**

evaluation_plots

This folder contains predicted-versus-actual temperature plots for the:	training set, validation,	test set. 
The scatter plots illustrate how closely the neural-network predictions match the original temperatures results obtained from Simufact Welding. The original temperature line is at 45 degrees angle and predictions by the model are shown in blue colour around the red original temperature line. The test-dataset scatter plots are important here as these are the predictions made by model on an unseen data after training.
____________________________________________________________________________________________________________________________










