import os, argparse, logging, pickle, json
import numpy as np
from tqdm import tqdm

# this is to validate that all model with the same dataset should have the same id2vocab?
logger = logging.getLogger("validate id2vocab")
logging.basicConfig(
    filename="logger.out",
    level=logging.INFO)

def dict_diff(dict1, dict2):
    only_in_1 = {k: dict1[k] for k in dict1 if k not in dict2}
    only_in_2 = {k: dict2[k] for k in dict2 if k not in dict1}
    different_values = {
        k: (dict1[k], dict2[k]) for k in dict1 if k in dict2 and dict1[k] != dict2[k]
    }
    return only_in_1, only_in_2, different_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root_dir", type=str, default="/data/tlippin1/tlippin1/detm-shootout/work")
    args = parser.parse_args()

    trials = os.listdir(args.root_dir)
    for trial_idx, trial in enumerate(trials):
        path_0 = os.path.join(args.root_dir, trial)
        idx2vocab = None
        logger.info(f"navigating on trial #{trial_idx}/{len(trials)} {trial}")
        if os.path.isdir(path_0) and "xDETM" in os.listdir(path_0):
            path_1 = os.path.join(path_0, "xDETM")
            subfolders = os.listdir(path_1)
            for folder_idx, subfolder in enumerate(subfolders):
                logger.info(f"navigating on folder #{folder_idx}/{len(subfolders)} {subfolder}")
                path_2 = os.path.join(path_1, subfolder)
                data = [file for file in os.listdir(path_2) if file.endswith("_hypercube.pkl.gz")]
                for data_instance in data:
                    path_3 = os.path.join(path_2, data_instance)
                    with open(path_3, "rb") as f:
                        pickle_instance = pickle.load(f)
                    if not idx2vocab:
                        idx2vocab = pickle_instance['index_to_word']
                        logger.info(f"setting idx2vocab for trial {trial} subfolder {subfolder} data instance {data_instance}")
                    else:
                        diff1, diff2, diff3 = dict_diff(idx2vocab, pickle_instance['index_to_word'])
                        assert not len(diff1) and not len(diff2) and not len(diff3)
        else:
            logger.info(f"trial {trial} does not have xDETM")