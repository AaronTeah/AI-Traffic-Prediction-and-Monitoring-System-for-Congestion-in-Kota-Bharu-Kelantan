import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Ensure the directory for saving models exists
os.makedirs('model', exist_ok=True)

# Load the dataset
df = pd.read_csv('traffic.csv')

# Handle missing values if any
df = df.dropna()

# Convert Time to a usable format (seconds since midnight)
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
df['Time'] = df['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Encode categorical variables
label_encoder_day = LabelEncoder()
label_encoder_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df['Day of the week'] = label_encoder_day.transform(df['Day of the week'])

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Time', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']])

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

# Define the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dropout(0.2))
model.add(Dense(1))  # Single output unit for regression

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('model/traffic_lstm_model.h5')

print(f'Test MAE: {np.mean(history.history["val_mae"])}')
