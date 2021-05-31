import sys


import numpy as np
from nnmnkwii.io import hts
from os.path import join
from glob import glob
from tqdm import tqdm


paths = sorted(glob(join("../data", "label_phone_align", "*.lab")))

for i, filepath in tqdm(enumerate(paths)):
    #print(filepath)
    label = hts.load(filepath)

    end_times_ = label.end_times
    end_times = []

    for j, end_time in enumerate(end_times_):
        str_tmp = str(label[j][2])
        if (
            "-a+" in str_tmp
            or "-i+" in str_tmp
            or "-u+" in str_tmp
            or "-e+" in str_tmp
            or "-o+" in str_tmp
            or "-cl+" in str_tmp
            or "-N+" in str_tmp
            or "-A+" in str_tmp
            or "-I+" in str_tmp
            or "-U+" in str_tmp
            or "-E+" in str_tmp
            or "-O+" in str_tmp
        ):
            #print(str_tmp[str_tmp.find('-')+1: str_tmp.find('+')])
            end_times.append(end_time - 1)
    #print(len(end_times))
    #print('\n\n\n')

    end_index = np.array(end_times) / 50000 - 1

    mora_index = np.array([0] * int(label[-1][1] / 50000))

    mora_index = [
        1 if i in list(end_index.astype(int)) else 0
        for i in range(int(label[-1][1] / 50000))
    ]
    #print(np.array(mora_index).sum())
    indices = label.silence_frame_indices().astype(int)

    #indices = indices[indices < len(mora_index)]
    mora_index = np.delete(mora_index, indices, axis=0)
    mora_index = mora_index.nonzero()[0] + 2

    np.savetxt(
        "../data/mora_index/squeezed_mora_index_"
        + "0" * (4 - len(str(i + 1)))
        + str(i + 1)
        + ".csv",
        mora_index,
    )
