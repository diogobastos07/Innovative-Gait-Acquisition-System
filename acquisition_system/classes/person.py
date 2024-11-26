import numpy as np
from sklearn.decomposition import PCA


class Person:
    def __init__(self, id, first_box):
        # Initialize person object with ID and first bounding box
        self.id = id
        self.box_history = [] # Stores the history of positions (bounding boxes)
        self.speed_history = [] # Stores the calculated speeds
        self.trendline = None # Stores the trendline coefficients
        self.box_history.append(first_box.astype(int).tolist()) # Add the initial box



    def add_position(self, box):
        # Add a new position (bounding box) to the history
        self.box_history.append(box.astype(int).tolist())


    def calculate_trendline_coefficients(self):
        # Calculate the coefficients of the trendline using PCA
        positions_history = np.array(self.box_history)[:, :2] # Extract positions (x, y)
        mean = np.mean(positions_history, axis=0) # Calculate the mean of positions
        data_centered = positions_history - mean # Center the data
        pca = PCA(n_components = 1) # Use PCA to find the principal direction
        pca.fit(data_centered)
        direction = pca.components_[0] # Get the principal direction vector
        # Calculate the slope and intercept of the trendline
        if direction[0] == 0:
            slope = 999999 # Handle undefined slope (vertical line)
        else:
            slope = direction[1] / direction[0]
        intercept = mean[1] - slope * mean[0]
        self.trendline = (slope, intercept)


    def has_variance(self):
        # Check if there is variance in the x or y coordinates
        var_x = np.var(np.array(self.box_history)[:, 0]) # Variance in x
        var_y = np.var(np.array(self.box_history)[:, 1]) # Variance in y
        return var_x != 0 or var_y != 0 # Return True if variance exists


    def distance_point_to_trendline(self):
        # Calculate the distance of the latest point to the trendline
        return abs(self.trendline[0] * self.box_history[-1][0] - self.box_history[-1][1] + self.trendline[1]) / np.sqrt(self.trendline[0]**2 + 1)
    
    
    def calculate_speed(self, x):
        # Calculate relative speed over the last 'x' positions
        history_last_x_boxes = np.array(self.box_history[-x:]) # Get the last 'x' boxes
        average_box_height = np.mean(history_last_x_boxes[:, 3]) # Average height of the bounding boxes
        # Calculate relative speed in x and y directions
        relative_speed_x = (history_last_x_boxes[-1][0] - history_last_x_boxes[0][0])/average_box_height
        relative_speed_y = (history_last_x_boxes[-1][1] - history_last_x_boxes[0][1])/average_box_height
        self.speed_history.append((relative_speed_x , relative_speed_y)) # Append to speed history

    
    def calculate_average_speed(self, t):
        # Calculate the average speed over the last 't' speed measurements
        history_last_t_speeds = np.array(self.speed_history[-t:]) # Get the last 't' speeds
        speeds = np.sqrt(np.sum(history_last_t_speeds**2, axis=1)) # Calculate magnitudes of speeds
        return np.mean(speeds) # Return the average speed




    