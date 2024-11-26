class LastFrames:
    def __init__(self, max_frames = 75):
        self.max_frames = max_frames # Maximum number of frames to store 'n'
        self.frames = [] # List to store the frames

    def add_frame(self, frame):
        # Add a new frame to the list
        if len(self.frames) == self.max_frames:
            self.frames.pop(0) # Remove the oldest frame if the limit is reached
        self.frames.append(frame) # Add the new frame to the end of the list

    def check_frame(self, num_max_frames):
        # Trim the list of frames to a specified maximum length
        if len(self.frames) > num_max_frames:
            self.frames = self.frames[len(self.frames)-num_max_frames:] # Keep only the most recent and potentially necessary frames