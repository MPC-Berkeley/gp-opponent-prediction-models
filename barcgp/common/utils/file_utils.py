import os
import pickle
import pathlib

gp_dir = os.path.expanduser('~') + '/barc_data/'
train_dir = os.path.join(gp_dir, 'trainingData/')
eval_dir = os.path.join(gp_dir, 'evaluationData/')
model_dir = os.path.join(train_dir, 'models')


def dir_exists(path=''):
    dest_path = pathlib.Path(path).expanduser()
    return dest_path.exists()


def create_dir(path='', verbose=False):
    dest_path = pathlib.Path(path).expanduser()
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
        return dest_path
    else:
        if verbose:
            print('- The source directory %s does not exist, did not create' % str(path))
        return None


def pickle_write(data, path):
    dbfile = open(path, 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()


def pickle_read(path):
    dbfile = open(path, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data
