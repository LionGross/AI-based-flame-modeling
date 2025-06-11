import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor
from itertools import product

#################################################### Load dataC:\Users\lg49jeki
#Data = pd.read_excel('W:/New_Data_Sven/Regressors_Average_Length_Width_Area/Parameters_AveragedLengthWidthArea.xlsx')
Data = pd.read_excel('C:/Users/lg49jeki/Parameters_AveragedLengthWidthArea_Modified.xlsx')
# Shuffle the data
Data = shuffle(Data, random_state=40)

# Input and output column names
input_columns = ['Power', 'Gas', 'Tangential', 'Axial']
output_columns = ['Flame Width AveragePixel', 'Flame Length Average Pixel', 'Flame Area Average Pixel^2']

# Extract features (X) and target (Y)
DataX = Data[input_columns].to_numpy()
DataY = Data[output_columns].to_numpy()

# Train-test split with indices tracking
XTrain, XTest, YTrain, YTest, train_indices, test_indices = train_test_split(
    DataX, DataY, np.arange(len(Data)), test_size=0.2, random_state=20
)
# Retrieve the test dataset using indices
TestData = Data.iloc[test_indices]
print("Test Dataset:")
print(TestData)



# Scale each column separately using Min-Max scaling
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

XTrain = scaler_X.fit_transform(XTrain)
XTest = scaler_X.transform(XTest)
YTrain = scaler_Y.fit_transform(YTrain)
YTest = scaler_Y.transform(YTest)

##################################################### List of models to evaluate
AllModels = [
    {'Name': 'Linear Regression', 'Regressor': LinearRegression()},
    {'Name': 'Ridge Regression', 'Regressor': Ridge()},
    {'Name': 'Lasso Regression', 'Regressor': Lasso()},
    {'Name': 'K Neighbors Regressor', 'Regressor': KNeighborsRegressor()},
    {'Name': 'Decision Tree Regressor', 'Regressor': DecisionTreeRegressor()},
    {'Name': 'Random Forest Regressor', 'Regressor': RandomForestRegressor()},
    {'Name': 'Gradient Boosting Regressor', 'Regressor': GradientBoostingRegressor()},
    {'Name': 'Adaboost Regressor', 'Regressor': AdaBoostRegressor()},
]

# Evaluate models
ModelNames = []
All_RMSE = []
All_RSquare = []

for Model in AllModels:
    ModelName = Model['Name']
    Regressor = MultiOutputRegressor(Model['Regressor'])  # Wrap the regressor
    ModelNames.append(ModelName)
    
    Regressor.fit(XTrain, YTrain)
    YPred = Regressor.predict(XTest)
    
    RMSE = np.sqrt(mean_squared_error(YTest, YPred))
    RSquare = r2_score(YTest, YPred, multioutput='uniform_average')  # Averaged R²
    
    All_RMSE.append(RMSE)
    All_RSquare.append(RSquare)

Results = pd.DataFrame({'ModelName': ModelNames, 'RMSE': All_RMSE, 'R Squared': All_RSquare})
print("All Models Score: \n")
print(Results.sort_values(by=['RMSE']))

##################################################### Train 
Regressor = RandomForestRegressor()
Regressor.fit(XTrain, YTrain)


##################################################### Predictions on test set with labels
# Predictions on test set
Predictions = Regressor.predict(XTest)

# Scale back predictions and YTest to original values
YTest_original = scaler_Y.inverse_transform(YTest)
Predictions_original = scaler_Y.inverse_transform(Predictions)

# Labels for plotting
output_labels = output_columns
video_names = TestData["Names"].values  # Get video names from the test set

for i in range(3):
    plt.figure(figsize=(12, 6), facecolor='w')  # Create a new figure for each variable
    
    # Scatter plot of actual vs. predicted values (scaled back)
    plt.scatter(range(len(YTest_original)), YTest_original[:, i], color='blue', label='Actual', s=100)
    plt.scatter(range(len(Predictions_original)), Predictions_original[:, i], color='red', label='Predicted', s=100)
    
    # Annotate video names
    for j in range(len(video_names)):
        plt.text(j, YTest_original[j, i], video_names[j], fontsize=10, ha='right', color='black', rotation=45)

    plt.title(f"{output_labels[i]}", fontsize=20)
    plt.xlabel("Sample Index", fontsize=15)
    plt.ylabel(output_labels[i], fontsize=15)
    plt.legend(loc='best')

    plt.show()


##################################################### Predictions on test set
Predictions = Regressor.predict(XTest)
Predictions = scaler_Y.inverse_transform(Predictions)  # Inverse transform to original scale

# Define input lists
power_list = [9]
fuel_list = [1]
tangential_list = [140] #[60, 70, 70, 90, 100, 110, 120, 130, 150, 160, 180, 200]
axial_list = [10] #[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

# Generate all possible combinations of inputs
input_combinations = list(product(power_list, fuel_list, tangential_list, axial_list))

# Convert to NumPy array
new_inputs = np.array(input_combinations)

# Scale the new inputs
new_inputs_scaled = scaler_X.transform(new_inputs)

# Predict flame properties for each input
predictions = Regressor.predict(new_inputs_scaled)
predictions = scaler_Y.inverse_transform(predictions)  # Inverse transform to original scale

# Create a DataFrame for better readability
results_df = pd.DataFrame(new_inputs, columns=['Power', 'Gas', 'Tangential', 'Axial'])
results_df[['Flame Width', 'Flame Length', 'Flame Area']] = predictions

# Display results
print(results_df)