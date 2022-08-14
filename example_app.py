import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.use("TKAgg")

def main(ref: str, signal: str = None, model=None, window_offset=0.2):
    """
        Parameters:
            ref: str : a (path to a) json with the audio workout signal times
            signal: str: a (path to a) csv with the recorded workout
            model (with x.predict) : an object with a predict function that returns a list of predictions
    """

    with open(file=ref, mode='r') as ref:
        cues = json.load(ref)

    cuts = cues["cuts"]
    offset = cues["start"]
    cuts = [x - offset for x in cuts]
    detections_times = []
    if model:
        with open(file=signal, mode='r') as csv:
            # TODO extract and preprocess sensor data
            sequences = None
            detections = model.predict(sequences)
            detections_times = [x[0] * window_offset for x in enumerate(detections) if x[1]]

    else:
        # create faux example
        detections_times = [x + 1*np.random.random() for x in cuts]
    data = np.concatenate((np.asarray([detections_times]), np.asarray([cuts])))
    fig = plt.eventplot(data, colors=["red", "blue"], lineoffsets=0)
    plt.xticks(np.arange(0, np.rint(data.max(axis=None))+0.2, step=0.2))
    # plt.tight_layout()
    plt.legend(["detection", "audio signal"], fontsize='xx-large')
    plt.title(f"Average reaction time: {np.round((np.asarray(detections_times) - np.asarray(cuts)).mean(), 2)}s", fontsize='xx-large')
    plt.yticks([])
    plt.ion()
    plt.show()
    plt.waitforbuttonpress()

if __name__ == "__main__":
    main(*sys.argv[1:])
