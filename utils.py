import pickle


class Logger():
    def __init__(self, **params):
        self.params = params
        self.data = []
        self.step_data = []

    def add_epoch_data(self, epoch, acc, loss):
        self.data.append({epoch: (acc, loss)})

    def add_step_data(self, step, acc, loss):
        self.step_data.append({step: (acc, loss)})

    def save(self, path):
        with open(path, 'wb') as logfile:
            pickle.dump(self, logfile)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as logfile:
            new_instance = pickle.load(logfile)
        return new_instance


def generate_model_checkpoint_name(stage, n_frames, ms_block=False, loss=None, optional=''):
    name = ""
    if stage < 3:
        name += RGB_PREFIX
        if stage == 2:
            name += '_stage2'
    elif stage == 3:
        name += FLOW_PREFIX
    else:
        name += JOINT_PREFIX
    name += '_' + str(n_frames) + 'frames'
    if loss is not None:
        name += '_' + loss
    if ms_block:
        name += '_msblock'
    name += optional + ".pth"

    return name


def generate_log_filenames(stage, n_frames, ms_block=False, loss=None, optional=''):
    train = LOG_PREFIX + str(stage) + '_' + str(n_frames) + 'frames'
    val = VAL_LOG_PREFIX + str(stage) + '_' + str(n_frames) + 'frames'
    if loss is not None:
        train += '_' + loss
        val += '_' + loss
    if ms_block:
        train += '_msblock'
        val += '_msblock'
    train += optional + ".obj"
    val += optional + ".obj"

    return train, val