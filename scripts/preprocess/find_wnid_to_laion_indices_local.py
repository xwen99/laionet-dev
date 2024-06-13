import sys
import os
import argparse
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--index2laionindices_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'ilsvrc_labels', 'icimagename2laionindices_top50.pkl'))

    parser.add_argument('--index2sims_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'ilsvrc_labels', 'icimagename2sims_top50.pkl'))

    parser.add_argument('--index2wnid_path', type=str,
                        default=os.path.join('imagenet-captions', 'processed', 'labels', 'icimagename2wnid.pkl'))

    # parser.add_argument('--sim_threshold', type=float, default=0.65)

    parser.add_argument('--save_file_name', type=str,
        default=f'wnid2laionindices({configs.LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_PREFIX[:-1]})_top50.pkl')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Save file path
    save_path = os.path.split(params['index2laionindices_path'])[0]

    print_verbose('done!\n')

    # ----- Load labels (maps) -----
    print_verbose('loading labels ...')

    with open(params['index2laionindices_path'], 'rb') as f:
        index2laionindices = pickle.load(f)

    with open(params['index2sims_path'], 'rb') as f:
        index2sims = pickle.load(f)

    with open(params['index2wnid_path'], 'rb') as f:
        index2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Mapping -----
    print_verbose('mapping ...')

    wnid2laionindices, passed_indices = {}, set()
    for idx, laion_indices, sims in zip(index2laionindices.keys(), index2laionindices.values(), index2sims.values()):
        wnid = index2wnid[idx]

        if wnid not in wnid2laionindices:
            wnid2laionindices[wnid] = []
        
        for laion_indice, sim in zip(laion_indices, sims):
            if laion_indice in passed_indices:
            # if laion_indice in wnid2laionindices[wnid]:
                continue
            # if sim >= params['sim_threshold']:
            wnid2laionindices[wnid].append(laion_indice)
            passed_indices.add(laion_indice)
            break

        # wnid2laionindices[wnid].extend(laion_indices)

    # Drop duplicate LAION index per WNID
    print_verbose(f'before: {sum([len(v) for v in wnid2laionindices.values()])}')
    wnid2laionindices = {k: list(set(v)) for k, v in wnid2laionindices.items()}
    print_verbose(f'after: {sum([len(v) for v in wnid2laionindices.values()])}')

    print_verbose('done!\n')

    # ----- Save -----
    print_verbose('saving ...')

    with open(os.path.join(save_path, params['save_file_name']), 'wb') as f:
        pickle.dump(wnid2laionindices, f)

    print_verbose('done!\n')
