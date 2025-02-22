from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

class EventsDataLoader:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.event_log_name = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')][0]
        self.event_accumulator = EventAccumulator(os.path.join(log_dir, self.event_log_name))
        self.event_accumulator.Reload()

    def get_scalars(self, tag):
        scalar_events = self.event_accumulator.Scalars(tag)
        scalar_values = [event.value for event in scalar_events]
        scalar_steps = [event.step for event in scalar_events]
        return scalar_steps, scalar_values

    def plot_scalars(self, tag, save=False):
        plt.clf()  # clear the current figure
        scalar_steps, scalar_values = self.get_scalars(tag)
        plt.plot(scalar_steps, scalar_values)
        plt.xlabel('Step')
        plt.ylabel(tag)
        plt.show()
        if save:
            _n = tag.replace('/', '_')
            plt.savefig(os.path.join(self.log_dir, _n + '.png'))

    def save_img(self, tag):
        plt.clf()   # clear the current figure
        scalar_steps, scalar_values = self.get_scalars(tag)
        plt.plot(scalar_steps, scalar_values)
        plt.xlabel('Step')
        plt.ylabel(tag)
        _n = tag.replace('/', '_')
        plt.savefig(os.path.join(self.log_dir, _n + '.png'))