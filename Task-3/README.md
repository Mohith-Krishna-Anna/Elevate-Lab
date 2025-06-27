I’m just getting started with machine learning and this helped me learn how to:

Train a basic ML model
Use sklearn and pandas
Plot a regression line
Measure how good the model is

Dataset Used

I used a file called Housing.csv which contains information about different houses like:

area (sq ft)
price
bedrooms
bathrooms
and more...
But in this project, I only used area to predict the price.

These are the Python libraries I used:

pandas → for reading and handling the data
sklearn → to train the regression model
matplotlib → for drawing the graph
What I Did (Steps)

Loaded the data from Housing.csv
Picked the area column as input (X), and price as output (y)
Split the data into training and testing sets (80% train, 20% test)
Trained a linear regression model using sklearn
Predicted house prices using the model
Calculated how accurate the model is using:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R² Score
Plotted the regression line so it’s easier to understand visually
