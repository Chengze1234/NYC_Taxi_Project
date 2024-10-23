# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:58:32 2024

@author: 24375
"""

# train_model.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import raw_taxi_df, clean_taxi_df, split_taxi_data
from model_utils import NYCTaxiExampleDataset, MLP

def main():
    """Main training script"""
    # Set fixed random number seed
    torch.manual_seed(42)
    # Load and clean the data
    raw_df = raw_taxi_df(filename="yellow_tripdata_2024-01.parquet")
    clean_df = clean_taxi_df(raw_df=raw_df)

    # Split the data into training and testing sets
    location_ids = ['PULocationID', 'DOLocationID']
    X_train, X_test, y_train, y_test = split_taxi_data(
        clean_df=clean_df,
        x_columns=location_ids,
        y_column="fare_amount",
        train_size=500000
    )

    # Create Dataset and DataLoader
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # Initialize MLP model
    mlp = MLP(encoded_shape=dataset.X_enc_shape)

    # Loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(5):  # 5 epochs
        print(f'Starting epoch {epoch + 1}')
        current_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            if i % 10 == 0:
                print(f'Loss after mini-batch {i + 1}: {current_loss / 500}')
            current_loss = 0.0

    print('Training complete.')

if __name__ == "__main__":
    main()
