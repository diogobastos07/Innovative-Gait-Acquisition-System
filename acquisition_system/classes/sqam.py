from .diagram import Diagram
from .last_frames import LastFrames
from .person import Person
import numpy as np
import math
from utils import get_msg_mgr
import os

class SQAM:
    def __init__(self, height, width, n = 75, p = 10, x = 5, t = 3, d = 15, v = 0.025, camera_dist = 920, diagrams = False):
        self.height = height
        self.width = width
        self.n = n # Maximum frames to track
        self.p = p # Minimum frames required for direction checks
        self.x = x # Frequency of speed checks and calculation
        self.t = t # Number of speed values used for average speed calculation
        self.d = d # Maximum allowed deviation for direction
        self.v = v # Minimum allowed average speed
        self.camera_dist = camera_dist # Distance from camera to tracking plane to angle calculation
        self.diagrams = diagrams # Whether to use diagrams for data visualization

        # Validate constraints
        if not (2 <= self.p < self.n):
            raise ValueError(f"Value of 'p' must be between 2 and 'n'={self.n} (exclusive).")
        if not (1 <= self.t):
            raise ValueError("Value of 't' must be greater than or equal to 1.")
        if not (2 <= self.x <= self.n / 2):
            raise ValueError(f"Value of 'x' must be between 2 and 'n/2'={self.n / 2}.")
        if not (self.x * self.t <= self.n):
            raise ValueError(f"The product of 'x' and 't' must be less than or equal to 'n'={self.n}.")
        if not (1 <= self.camera_dist):
            raise ValueError("Value of 'camera_dist' must be greater than or equal to 1.")

        # Initialize internal data structures
        self.people = [] # List of tracked people
        self.last_frames = LastFrames(self.n) # Store last frames with maximum of 'n' last frames
        if self.diagrams:
            self.all_data = Diagram() # For all data points
            self.filtered_data = Diagram() # For valid data points (complete sequences that meet criteria)



    def add_new_people(self, boxes, track_ids):
        # Add new people to tracking system
        for i, id in enumerate(track_ids):
            person = Person(id, boxes[i])
            if self.diagrams:
                self.all_data.add_point(id, boxes[i].astype(int).tolist()) # Add to diagram
            self.people.append(person)
            self.new_entries_num += 1



    def delete_person(self, person, reason = None):
        # Remove a person from tracking and log the reason if provided
        if reason:
            self.exclusion_dict.append({
                "person_id": person.id,
                "reason": reason,
                "frames_tracked": len(person.box_history),
                "first_position": person.box_history[0][:2],
                "last_position": person.box_history[-1][:2]
            })
        self.people.remove(person)



    def process_new_frame(self, frame, boxes, track_ids):
        # Process a new video frame and update tracking information
        self.last_frames.add_frame(frame) # Add frame to history
        num_max_frames = 0
        self.new_entries_num = 0
        detections_num = len(track_ids)
        self.exclusion_dict = [] # To store excluded sequences information
        self.complete_sequence_dict = [] # To store valid sequences information

        if not self.people:
            # If no people are being tracked, add new ones
            self.add_new_people(boxes, track_ids)
            if track_ids:
                num_max_frames = 1
        else:
            people_copy = self.people.copy()
            for person in people_copy:
                if person.id in track_ids:
                    # Update existing person's tracking data
                    idx = track_ids.index(person.id)
                    person.add_position(boxes[idx])
                    valid = True
                    num_frames_tracked = len(person.box_history)

                    # Perform validation checks
                    if num_frames_tracked >= self.p:
                        valid = self.check_direction_changes(person)
                    if valid:
                        if num_frames_tracked % self.x == 0:
                            valid = self.check_minimum_speed(person)
                            if valid:
                               valid = self.check_direction_reversal(person)

                    if valid:
                        # Update diagrams and process valid sequences
                        if self.diagrams:
                            self.all_data.add_point(person.id, boxes[idx].astype(int).tolist())
                        if num_frames_tracked > num_max_frames:
                            num_max_frames = num_frames_tracked
                        if num_frames_tracked == self.n:
                            angle = self.get_angle(person.box_history, person.trendline)
                            if self.diagrams:
                                self.filtered_data.add_set(person.id, person.box_history, angle)
                                self.filtered_data.add_point(0, boxes[idx].astype(int).tolist())
                            self.use_valid_data(person, angle)
                            self.delete_person(person)
                    else:
                        if self.diagrams:
                            self.all_data.add_point(0, boxes[idx].astype(int).tolist())
                    del track_ids[idx]
                    boxes = np.delete(boxes.astype(int).tolist(), idx, axis=0)
                else:
                    # Remove person if tracking is lost
                    self.delete_person(person, "Tracking Discontinuity")

            # Add new people to the system
            if len(track_ids) > 0:
                self.add_new_people(boxes, track_ids)
                if num_max_frames == 0:
                    num_max_frames = 1

        # Update frame tracking and tracking information
        self.last_frames.check_frame(num_max_frames)
        self.tracking_dict = {
                "detections_num": detections_num,
                "new_entries": self.new_entries_num,
                "exclusions_num": len(self.exclusion_dict),
                "valid_sequence_num": len(self.complete_sequence_dict),
                "num_max_frames": num_max_frames
            }



    def check_direction_changes(self, person):
        # Check if the person's direction remains consistent
        if len(person.box_history) > self.p:
            distance = person.distance_point_to_trendline()
            if distance > self.d:
                self.delete_person(person, f"Direction Change ({distance:.2f}>{self.d})")
                return False
        else:
            if not person.has_variance():
                self.delete_person(person, "Zero Variance")
                return False
        person.calculate_trendline_coefficients()
        return True



    def check_minimum_speed(self, person):
        # Ensure the person maintains a minimum speed
        person.calculate_speed(self.x)
        if len(person.speed_history) >= self.t:
            speed = person.calculate_average_speed(self.t)
            if speed < self.v:
                self.delete_person(person, f"Below Minimum Speed ({speed:.4f}<{self.v})")
                return False
        return True



    def check_direction_reversal(self, person):
        # Detect reversal in direction based on speed history
        if len(person.speed_history) >= 2:
            if (person.speed_history[-1][0]*person.speed_history[-2][0] < 0) and (person.speed_history[-1][1]*person.speed_history[-2][1] < 0):
                self.delete_person(person, "Reversed Direction")
                return False
        return True
    


    def get_angle(self, box_history, trendline):
        # Calculate the angle between the trajectory and a reference vector
        camera_position = [int(self.width/2), self.height + self.camera_dist]
        if (camera_position[0]-box_history[int(self.n/2)][0]) != 0:
            slope = (camera_position[1]-box_history[int(self.n/2)][1])/(camera_position[0]-box_history[int(self.n/2)][0])
            if slope > 0:
                vector_director_1 = np.array([1, slope])
            else:
                vector_director_1 = np.array([-1, -slope])
            if trendline[0] > 0:
                if box_history[0][1] < box_history[self.n - 1][1]:
                    vector_director_2 = np.array([1, trendline[0]])
                else:
                    vector_director_2 = np.array([-1, -trendline[0]])
            else:
                if box_history[0][1] < box_history[self.n - 1][1]:
                    vector_director_2 = np.array([-1, -trendline[0]])
                else:
                    vector_director_2 = np.array([1, trendline[0]]) 
            dot_product = np.dot(vector_director_1, vector_director_2)
            norm_1 = np.linalg.norm(vector_director_1)
            norm_2 = np.linalg.norm(vector_director_2)
            cos_angle = dot_product / (norm_1 * norm_2)
            angle_radians = math.acos(cos_angle)
            angle_degrees = math.degrees(angle_radians)
            if vector_director_1[0] == 1:
                if (trendline[0] < 0 and vector_director_2[0] == 1) or (trendline[0] > 0 and ((vector_director_2[0] == -1 and vector_director_1[1] < abs(vector_director_2[1])) or (vector_director_2[0] == 1 and vector_director_1[1] > vector_director_2[1]))):
                    angle_degrees = 360 - angle_degrees
            else:
                if (trendline[0] > 0 and vector_director_2[0] == 1) or (trendline[0] < 0 and ((vector_director_2[0] == 1 and vector_director_1[1] > vector_director_2[1]) or (vector_director_2[0] == -1 and vector_director_1[1] < abs(vector_director_2[1])))):
                    angle_degrees = 360 - angle_degrees
        else:
            if box_history[0][1] < box_history[self.n - 1][1]:
                angle_degrees = 0
            else:
                angle_degrees = 180
        return int(angle_degrees)
    


    def use_valid_data(self, person, angle):
        # Store or process valid sequences with calculated angle
        # DO WHAT YOU WANT WITH VALID DATA USING "person", "angle" AND "self.last_frames"
        self.complete_sequence_dict.append({
            "person_id": person.id,
            "angle": angle,
            "frames_tracked": len(person.box_history),
            "first_position": person.box_history[0][:2],
            "last_position": person.box_history[-1][:2]
        })



    def end(self, output_path):
        # Save diagrams and log the completion of processing
        msg_mgr = get_msg_mgr()
        if self.diagrams:
            output_path = os.path.join(output_path, 'diagrams')
            msg_mgr.log_info(f"Diagrams and respective legends are saved in {output_path}")
            self.all_data.save_diagram(True, self.height, self.width, output_path)
            self.filtered_data.save_diagram(False, self.height, self.width, output_path)
        msg_mgr.log_info('IT\'S FINISH!')
        output_path_log = os.path.join(output_path, 'logs')
        print(f'Log file is saved in {output_path_log}')