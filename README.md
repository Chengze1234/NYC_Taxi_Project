# NYC Taxi Trip Fare Prediction

## Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

To export the dependencies, use:

```bash
pip freeze > requirements.txt
```

## How to Run the Code

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/NYC_Taxi_Project.git
   cd NYC_Taxi_Project
   ```

2. **Download the dataset**:

   Place the dataset `yellow_tripdata_2024-01.parquet` in the `data/` folder.
   This can be done by:
    ```bash
  bash get_taxi.sh
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the training script**:

   To load the data, preprocess it, and train the model, run:

   ```bash
   python train_model.py
   ```

## Project Structure

```
NYC_Taxi_Project/
├── data/                       # Folder for datasets
│   └── yellow_tripdata_2024-01.parquet
├── models/                     # Folder for saving model files
├── scripts/
│   ├── data_utils.py           # Functions for data loading, cleaning, splitting
│   └── model_utils.py          # Dataset class, model architecture, training code
├── train_model.py              # Main script to load data, train model
├── README.md                   # Project description and how to run the code
└── requirements.txt            # Python dependencies
```
