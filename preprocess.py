import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# Load the dataset
df = pd.read_csv('traffic.csv')

# Handle missing values if any
df = df.dropna()

# Convert Time to a usable format (seconds since midnight)
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
df['Time'] = df['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Day of the week'] = label_encoder.fit_transform(df['Day of the week'])
df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop(columns=['Date', 'Traffic Situation']))

# Prepare the final dataset for model training
X = []
y = []

time_steps = 15  # 15-minute time steps

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i])
    y.append(df['Traffic Situation'].iloc[i])

X, y = np.array(X), np.array(y)

# Save label encoder and scaler for future use
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
