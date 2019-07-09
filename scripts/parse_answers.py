import os
import pickle
import sys

if __name__ == "__main__":
    labels, files = pickle.load(open(sys.argv[1], 'rb'))

    for x in range(16):
        os.mkdir(str(x))

    for i, key in enumerate(files):
        from_ = os.path.join('./test', key)
        to_ = labels[i]
        os.system('cp {} {}'.format(from_, to_))
