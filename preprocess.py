import dill
from utils_gan import *
import sys
import os


def main(path):
    indexer = Indexer()
    filenames = ['train_source.txt', 'train_target.txt', 'test_source.txt', 'test_target.txt']
    for filename in filenames:
        indexer.add_document(os.path.join(path, filename), True)

    if not os.path.isdir(os.path.join(path, 'info')):
        os.mkdir(os.path.join(path, 'info'))

    dill.dump(indexer, open(os.path.join(path, 'info', "indexer2.p"), "wb"))



if __name__ == '__main__':
    main(sys.argv[1])
