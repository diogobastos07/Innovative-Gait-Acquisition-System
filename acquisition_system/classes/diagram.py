import matplotlib.pyplot as plt
import os
from utils import get_color_for_id
from matplotlib.gridspec import GridSpec

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
        # Configure the main plot
        fig, ax = plt.subplots(figsize=(12, 8))
        points_set_0 = self.points.pop(0, [])  # Separate points for set 0 (special moments)
        
        # Plot points for each set
        for (set, points) in self.points.items():
            x = [p[0] for p in points]  # X-coordinates
            y = [p[1] for p in points]  # Inverted Y-coordinates
            color = get_color_for_id(set)  # Get a unique color for the set ID
            if all_data:
                ax.scatter(x, y, label=set, color=tuple(c / 255.0 for c in color))  # Plot all data points
            else:
                angles_set = self.angles[set]
                angles_string = ", ".join(map(str, angles_set))  # Format angles as a string
                ax.scatter(x, y, label=f"{set}: {angles_string}", color=tuple(c / 255.0 for c in color))  # Plot filtered points with angles
        
        # Plot points for set 0 (black)
        if points_set_0:
            x = [p[0] for p in points_set_0]
            y = [p[1] for p in points_set_0]
            ax.scatter(x, y, label=0, color='black')
        
        # Set plot limits, labels, and title
        ax.set_xlim(0, max_width)
        ax.set_ylim(-max_height, 0)
        ax.set_xlabel('Coordinate X')
        ax.set_ylabel('Inverted Coordinate Y')
        if all_data:
            ax.set_title('Trajectory Plot of All Data Points by ID')
        else:
            ax.set_title('Trajectory Plot of Valid Data Points by ID and Respective Angles')
        ax.grid(True)

        # Create a separate legend figure
        handles, labels = ax.get_legend_handles_labels()  # Get legend handles and labels

        # Create a combined layout
        if all_data:
            combined_fig = plt.figure(figsize=(16, 8))
        else:
            combined_fig = plt.figure(figsize=(16, 16))
        
        if all_data:
            # Place the legend to the right of the plot
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=combined_fig)
            main_ax = combined_fig.add_subplot(gs[0, 0])  # Plot area
            legend_ax = combined_fig.add_subplot(gs[0, 1])  # Legend area
        else:
            # Place the legend below the plot
            gs = GridSpec(2, 1, height_ratios=[3, 1], figure=combined_fig)
            main_ax = combined_fig.add_subplot(gs[0, 0])  # Plot area
            legend_ax = combined_fig.add_subplot(gs[1, 0])  # Legend area

        # Recreate the main plot
        for (set, points) in self.points.items():
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            color = get_color_for_id(set)
            if all_data:
                main_ax.scatter(x, y, label=set, color=tuple(c / 255.0 for c in color))
            else:
                angles_set = self.angles[set]
                angles_string = ", ".join(map(str, angles_set))
                main_ax.scatter(x, y, label=f"{set}: {angles_string}", color=tuple(c / 255.0 for c in color))
        
        if points_set_0:
            x = [p[0] for p in points_set_0]
            y = [p[1] for p in points_set_0]
            main_ax.scatter(x, y, label=0, color='black')

        # Configure main_ax settings
        main_ax.set_xlim(0, max_width)
        main_ax.set_ylim(-max_height, 0)
        main_ax.set_xlabel('Coordinate X')
        main_ax.set_ylabel('Inverted Coordinate Y')
        main_ax.set_title(ax.get_title())
        main_ax.grid(True)

        # Recreate the legend
        legend_ax.axis('off')  # Turn off axes for the legend
        legend_ax.legend(handles, labels, loc='center', fontsize=10, ncol=3, title="Legend by ID" if all_data else "Legend by ID with Respective Angles")

        # Save the combined figure
        os.makedirs(output_path, exist_ok=True)
        if all_data:
            combined_fig.savefig(os.path.join(output_path, 'all_data.png'), format='png', bbox_inches='tight')
        else:
            combined_fig.savefig(os.path.join(output_path, 'filtered_data.png'), format='png', bbox_inches='tight')
        plt.close(combined_fig)
