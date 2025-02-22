import cv2


class Frames2Video:
    def __init__(self, video_name, fps=25, frame_width=1280, frame_height=720):
        self.video_name = video_name + ".mp4"
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.video_name, fourcc, self.fps, (frame_width, frame_height))

    def add_frame(self, frame):
        if frame is not None:
            if frame.shape[0] == self.frame_height and frame.shape[1] == self.frame_width:
                frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame)
            else:
                print("Frame size does not match the video size.")

    def close(self):
        self.video_writer.release()
        cv2.destroyAllWindows()
        print("Video saved as: ", self.video_name)

    def __del__(self):
        self.close()