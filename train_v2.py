import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import matplotlib.pyplot as plt

# Ensure the directory for saving models exists
os.makedirs('model', exist_ok=True)

# Load the dataset
df = pd.read_csv('traffic.csv')

# Handle missing values if any
df = df.dropna()

# Function to validate time format
def is_valid_time_format(time_str):
    try:
        pd.to_datetime(time_str, format='%I:%M:%S %p')
        return True
    except ValueError:
        return False

# Identify and print invalid time entries
invalid_times = df[~df['Time'].apply(is_valid_time_format)]
if not invalid_times.empty:
    print("Invalid time entries found:\n", invalid_times)

# Remove invalid time entries
df = df[df['Time'].apply(is_valid_time_format)]

# Convert Time to a usable format (seconds since midnight)
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
df['Time'] = df['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Feature engineering: Add hour and minute as separate features
df['Hour'] = pd.to_datetime(df['Time'], unit='s').dt.hour
df['Minute'] = pd.to_datetime(df['Time'], unit='s').dt.minute

# Encode categorical variables
label_encoder_day = LabelEncoder()
label_encoder_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df['Day of the week'] = label_encoder_day.transform(df['Day of the week'])

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Time', 'Hour', 'Minute', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']])

# Prepare the final dataset for model training
X = []
y = []

time_steps = 15  # 15-minute time steps

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i])
    y.append(df['Total'].iloc[i])

X, y = np.array(X), np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save label encoder and scaler for future use
joblib.dump(label_encoder_day, 'model/label_encoder_day.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Define the enhanced LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dropout(0.3))
model.add(Dense(1))  # Single output unit for regression

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('model/traffic_lstm_model.h5')

# Evaluate the model on the validation set
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation MAE: {val_mae}')

# Plot training & validation loss and MAE over epochs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot MAE
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Training and Validation MAE')
ax2.set_xlabel
