import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('coral_bleaching_data.csv')


# Convert columns to numeric types
data['Turbidity'] = pd.to_numeric(data['Turbidity'], errors='coerce')
data['Temperature_Maximum'] = pd.to_numeric(data['Temperature_Maximum'], errors='coerce')
data['Temperature_Minimum'] = pd.to_numeric(data['Temperature_Minimum'], errors='coerce')
data['Temperature_Mean'] = pd.to_numeric(data['Temperature_Mean'], errors='coerce')
data['Windspeed'] = pd.to_numeric(data['Windspeed'], errors='coerce')
data['Depth_m'] = pd.to_numeric(data['Depth_m'], errors='coerce')
data['Cyclone_Frequency'] = pd.to_numeric(data['Cyclone_Frequency'], errors='coerce')
data['Distance_to_Shore'] = pd.to_numeric(data['Distance_to_Shore'], errors='coerce')
data['ClimSST'] = pd.to_numeric(data['ClimSST'], errors='coerce')
data['Temperature_Kelvin'] = pd.to_numeric(data['Temperature_Kelvin'], errors='coerce')
data['Temperature_Kelvin_Standard_Deviation'] = pd.to_numeric(data['Temperature_Kelvin_Standard_Deviation'], errors='coerce')
data['Percent_Bleaching'] = pd.to_numeric(data['Percent_Bleaching'], errors='coerce')


# Drop rows with missing values
data = data.dropna()

# Define independent variables (X) and dependent variable (y)
X = data[['Turbidity', 'Temperature_Maximum', 'Temperature_Minimum', 'Temperature_Mean', 'Windspeed', 'Depth_m', 'Cyclone_Frequency', 'Distance_to_Shore', 'ClimSST', 'Temperature_Kelvin', 'Temperature_Kelvin_Standard_Deviation']]  # Replace with your actual column names
y = data['Percent_Bleaching']  # Replace with your actual column name

# Add a constant term to the independent variables matrix
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display the regression summary
print(model.summary())

# Predict the values using the model
y_pred = model.predict(X)

# Plot the actual vs predicted values with a regression line
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.regplot(x=y, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title('Actual vs Predicted Coral Bleaching')
plt.xlabel('Actual Percentage of Bleaching')
plt.ylabel('Predicted Percentage of Bleaching')

# Annotate the regression equation on the plot

# Annotate the regression equation on the plot
equation = (
    f'Regression Equation:\n y = {model.params[0]:.2f} + '
    f'{model.params[1]:.2f} * Factor1 + {model.params[2]:.2f} * Factor2 + '
    f'{model.params[3]:.2f} * Factor3 + {model.params[4]:.2f} * Factor4 + '
    f'{model.params[5]:.2f} * Factor5 + {model.params[6]:.2f} * Factor6 + '
    f'{model.params[7]:.2f} * Factor7 + {model.params[8]:.2f} * Factor8 + '
    f'{model.params[9]:.2f} * Factor9 + {model.params[10]:.2f} * Factor10 + '
    f'{model.params[11]:.2f} * Factor11'
)


plt.show()