# import packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import Sequential
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense

#   1-	Data Preparation   #

# Load the data
data = pd.read_csv('data.csv')
ages = pd.read_csv('Ages.csv')

# Merge the data and ages on the sample name
df = pd.merge(data, ages, on='Sample Accession')

# Drop sample_name as it's not needed for modeling
df.drop(columns=['Sample Accession'], inplace=True)

# Separate features and target
X = df.drop(columns=['Age'])
y = df['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  2. Exploratory Data Analysis (EDA)  #

# Distribution of ages
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#  3. Model Training and Evaluation  #

#  3.1 Random Forest Regressor  #

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f'Random Forest MAE: {rf_mae}')
print(f'Random Forest R-squared: {rf_r2}')

#  3.2 Gradient Boosting Regressor  #

# Initialize and train the model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on test data
gb_predictions = gb_model.predict(X_test)

# Evaluate the model
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)
print(f'Gradient Boosting MAE: {gb_mae}')
print(f'Gradient Boosting R-squared: {gb_r2}')

#  3.3 Neural Network (Deep Learning Model)  #

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
nn_model = Sequential()
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1))

# Compile the model
nn_model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
nn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on test data
nn_predictions = nn_model.predict(X_test_scaled).flatten()

# Evaluate the model
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)
print(f'Neural Network MAE: {nn_mae}')
print(f'Neural Network R-squared: {nn_r2}')

#  4. Results Visualization  #

plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_predictions, label='Random Forest', alpha=0.6)
plt.scatter(y_test, gb_predictions, label='Gradient Boosting', alpha=0.6)
plt.scatter(y_test, nn_predictions, label='Neural Network', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Predicted vs Actual Age')
plt.legend()
plt.show()
