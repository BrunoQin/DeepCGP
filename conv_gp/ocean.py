import numpy as np
import gzip
import pickle
from sklearn import preprocessing
from gpflow import settings
from arguments import default_parser, train_steps
from experiment import Experiment

def load_ocean():
    # read data
    with gzip.open('OCEAN_data/redata.pkl.gz') as fp:
        redata = np.array(pickle.load(fp)).astype(np.float32)

    with gzip.open('OCEAN_data/nino.pkl.gz') as fp:
        nino = np.array(pickle.load(fp)).astype(np.float32)

    redata = np.array(redata).astype(np.float32)
    nino = np.array(nino).astype(np.float32)
    X = redata[0:4600]
    Y = nino[11:4611]
    Xt = redata[4601:4789]
    Yt = nino[4612:4800]
    return X, Y, Xt, Yt


class Ocean(Experiment):
    def _load_data(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = load_ocean()
        self._select_training_points()
        self._select_test_points()
        self._preprocess_data()

    def _select_training_points(self):
        chosen = slice(0, self.flags.N)
        self.X_train = self.X_train[chosen, :]
        self.Y_train = self.Y_train[chosen, :]

    def _select_test_points(self):
        arange = np.arange(0, len(self.X_test))
        chosen = np.random.choice(arange, self.flags.test_size, replace=False)
        self.X_test = self.X_test[chosen, :]
        self.Y_test = self.Y_test[chosen, :]

    def _preprocess_data(self):
        self.X_transform = preprocessing.StandardScaler()
        self.X_train = self.X_transform.fit_transform(self.X_train).astype(settings.float_type)
        self.X_test = self.X_transform.transform(self.X_test).astype(settings.float_type)
        self.X_train = self.X_train.reshape(-1, 150, 160, 1)
        self.X_test = self.X_test.reshape(-1, 150, 160, 1)

def read_args():
    parser = default_parser()
    parser.add_argument('--tensorboard-dir', type=str, default='/tmp/ocean/tensorboard')
    parser.add_argument('--model-path', default='/tmp/ocean/model.npy')
    parser.add_argument('-N', type=int,
                        help="How many training examples to use.", default=4600)
    return parser.parse_args()


def main():
    flags = read_args()

    experiment = Ocean(flags)

    try:
        for i in range(train_steps(flags)):
            experiment.train_step()
    finally:
        experiment.conclude()


if __name__ == "__main__":
    main()


