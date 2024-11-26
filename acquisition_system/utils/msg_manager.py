import time
import logging
from time import strftime, localtime
import os


class MessageManager:
    def __init__(self):
        # Initialize time and iteration counter
        self.time = time.time()
        self.iteration = 0


    def init_logger(self, save_path, log_to_file):
        # Configure logging system
        self.logger = logging.getLogger('acquisition_system')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_file:
            os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
            vlog = logging.FileHandler(
                os.path.join(save_path, "logs", strftime('%Y-%m-%d-%H-%M-%S', localtime())+'.txt'))
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            self.logger.addHandler(vlog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.addHandler(console)


    def log_system_info(self, tracking_dict, exclusion_dict, complete_sequence_dict):
        # Log system state details
        now = time.time()
        self.iteration += 1
        tracking_info = ", ".join(f"{key}: {value}" for key, value in tracking_dict.items())
        header = f"Frame {self.iteration:05}, Cost {(now - self.time) * 1000:.2f}ms, {tracking_info}"
        if exclusion_dict:
            exclusions = "\n                      Excluded sequences:"
            for exclusion in exclusion_dict:
                exclusion_line = ", ".join(f"{key}: {value}" for key, value in exclusion.items())
                exclusions = f"{exclusions}\n                      -{exclusion_line}"
        else:
            exclusions = ""
        if complete_sequence_dict:
            completions = "\n                      Completed sequences:"
            for complete_sequence in complete_sequence_dict:
                complete_sequence_line = ", ".join(f"{key}: {value}" for key, value in complete_sequence.items())
                completions = f"{completions}\n                      -{complete_sequence_line}"
        else:
            completions = ""
        log_message = f"{header}{exclusions}{completions}"
        self.log_info(log_message)
        self.reset_time()


    def reset_time(self):
        self.time = time.time()


    def log_debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)


    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)


    def log_warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)


# Global MessageManager instance
msg_mgr = MessageManager()

def get_msg_mgr():
    return msg_mgr