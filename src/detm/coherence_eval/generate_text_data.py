import os, argparse, logging, pickle, json, gzip
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("get_corpus_data")
logging.basicConfig(level=logging.INFO)

def get_metainfo(field):
    if type(field) == np.ndarray:
        return f"{type(field)} <{field.shape}>"
    elif type(field) == dict:
        return f"{type(field)} <{len(field)}>"
    return f"{type(field)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root_dir", type=str, default="/data/tlippin1/tlippin1/detm-shootout/work")
    parser.add_argument("-o", "--output_dir", type=str, default="data")
    parser.add_argument("-n", "--top_n", type=int, default=100)
    args = parser.parse_args()

    trials = os.listdir(args.root_dir)
    for trial_idx, trial in enumerate(trials):
        path_0 = os.path.join(args.root_dir, trial)
        logger.info(f"navigating on trial #{trial_idx + 1}/{len(trials)} {trial}")
        if os.path.isdir(path_0) and "test_data.jsonl.gz" in os.listdir(path_0):
            path_1 = os.path.join(path_0, "test_data.jsonl.gz")

            with gzip.open(path_1, 'rt', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    print(data.keys())
                    print(len(data['content']))
                    print(len(data['content'][0]))
                    exit(0)