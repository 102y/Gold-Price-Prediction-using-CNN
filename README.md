# Gold-Price-Prediction-using-CNN
This project utilizes a Convolutional Neural Network (CNN) to predict future gold prices based on historical data. The model is trained on time series data to predict the next price, using a deep learning architecture specifically designed to capture patterns in sequential data.
Project Features
Data Preprocessing: The model preprocesses historical gold prices by removing commas, normalizing values, and creating time-series datasets for training and testing.
Model Architecture: The CNN model includes Conv1D, MaxPooling1D, Dense, and Dropout layers to effectively learn from the time series data.
Training: The model is trained on 80% of the dataset, and its performance is evaluated on the remaining 20% using the Mean Squared Error (MSE) metric.
Visualization: The project provides a comparison between the actual test data and predicted test data with a plot for better visualization of model performance.
Technologies Used
Python
TensorFlow/Keras
Scikit-learn
Matplotlib for visualization
How to Run
Clone the repository.
Install the required dependencies in requirements.txt.
Run the script untitled9.py to preprocess the data, train the model, and visualize predictions.
Future Enhancements
Implementing LSTM layers for more accurate sequential prediction.
Hyperparameter tuning for further optimization.
Expanding the model to predict other financial assets, such as stocks or cryptocurrencies.
