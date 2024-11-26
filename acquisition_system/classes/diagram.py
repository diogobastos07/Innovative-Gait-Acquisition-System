import matplotlib.pyplot as plt
import os
from utils import get_color_for_id

class Diagram: 
    def __init__(self):
        self.points = {} # Dictionary to store points grouped by set IDs
        self.angles = {} # Dictionary to store angles for each set of valid sequences


    def add_point(self, set, box):
        # Add a single point (bounding box) to a specific set
        if set not in self.points:
            self.points[set] = []
        self.points[set].append([box[0], -box[1]])


    def add_set(self, set, boxes, angle):
        # Add multiple points (bounding boxes) and an angle to a specific set
        if set not in self.points:
            self.points[set] = []
        for box in boxes:
            self.points[set].append([box[0], -box[1]])
        if set not in self.angles:
            self.angles[set] = []
        self.angles[set].append(angle)

    
    def save_diagram(self, all_data, max_height, max_width, output_path):
        # Save the diagram as a plot and generate a legend
        fig, ax = plt.subplots(figsize = (12, 8))
        points_set_0 = self.points.pop(0, []) # Separate points for set 0 (special moments)
        
        # Plot points for each set
        for (set, points) in self.points.items():
            x = [p[0] for p in points] # Extract x-coordinates
            y = [p[1] for p in points] # Extract inverted y-coordinates
            color = get_color_for_id(set) # Get unique color for the set ID
            if all_data:
                ax.scatter(x, y, label = set, color=tuple(c / 255.0 for c in color)) # Plot all data points
            else:
                angles_set = self.angles[set]
                angles_string = ", ".join(map(str, angles_set)) # Format angles as a string
                ax.scatter(x, y, label = f"{set}: {angles_string}", color = tuple(c / 255.0 for c in color)) # Plot filtered points with angles
        
        # Plot points for set 0
        if points_set_0:
            x = [p[0] for p in points_set_0]
            y = [p[1] for p in points_set_0]
            ax.scatter(x, y, label = 0, color = 'black') # Use black color for set 0

        # Configure plot limits and labels
        ax.set_xlim(0, max_width)
        ax.set_ylim(-max_height, 0)
        ax.set_xlabel('Coordinate X')
        ax.set_ylabel('Inverted Coordinate Y')

        # Set title based on the type of data being plotted
        if all_data:
            ax.set_title('Trajectory Plot of All Data Points by ID')
        else:
            ax.set_title('Trajectory Plot of Valid Data Points by ID and Respective Angles')
        
        ax.grid(True) # Enable grid on the plot
        os.makedirs(output_path, exist_ok = True) # Ensure the output directory exists

        # Save the plot to a file
        if all_data:
            fig.savefig(os.path.join(output_path, 'all_data.png'), format = 'png', bbox_inches = 'tight')
        else:
            fig.savefig(os.path.join(output_path, 'filtered_data.png'), format = 'png', bbox_inches = 'tight')
        plt.close(fig)
        
        # Generate and save a legend as a separate plot
        legend_fig, legend_ax = plt.subplots(figsize = (4, len(self.points) * 0.1))
        legend_ax.axis('off') # Hide axes for the legend
        handles, labels = ax.get_legend_handles_labels() # Get legend handles and labels
        
        if all_data:
            # Save legend for all data points
            legend_ax.legend(handles, labels, loc = 'center', fontsize = 10, ncol = 3, title = "Legend by ID")
            legend_fig.savefig(os.path.join(output_path, 'all_data_legend.png'), format='png', bbox_inches = 'tight')
        else:
            # Save legend for filtered points with angles
            legend_ax.legend(handles, labels, loc = 'center', fontsize = 10, ncol = 3, title = "Legend by ID with Respective Angles")
            legend_fig.savefig(os.path.join(output_path, 'filtered_data_legend.png'), format='png', bbox_inches='tight')
        plt.close(legend_fig) 