import os
import pickle

labels, files = pickle.load(
    open('labels-2019-07-05 17:29:19.010703.pickle', 'rb'))

for x in range(16):
    os.mkdir(str(x))

for i, key in enumerate(files):
    from_ = os.path.join('./test', key)
    to_ = labels[i]
    os.system('cp {} {}'.format(from_, to_))
