import os, argparse, logging, pickle, json
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("coherence analysis")
logging.basicConfig(
    # filename="coherence_analysis.out", 
    level=logging.INFO)

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
    parser.add_argument("-np", "--numpy_subdir", type=str, default="numpy_arr")
    parser.add_argument("-td", "--topdata_subdir", type=str, default="topic_data")
    parser.add_argument("-iv", "--idx2vocab_subdir", type=str, default="idx2vocab")
    parser.add_argument("-n", "--top_n", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.topdata_subdir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.numpy_subdir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.idx2vocab_subdir}", exist_ok=True)

    trials = os.listdir(args.root_dir)
    for trial_idx, trial in enumerate(trials):
        stored_idx2vocab = False
        path_0 = os.path.join(args.root_dir, trial)
        logger.info(f"navigating on trial #{trial_idx}/{len(trials)} {trial}")
        if os.path.isdir(path_0) and "xDETM" in os.listdir(path_0):
            path_1 = os.path.join(path_0, "xDETM")
            per_trial_json_data = {}
            subfolders = os.listdir(path_1)
            for folder_idx, subfolder in enumerate(subfolders):
                logger.info(f"navigating on folder #{folder_idx}/{len(subfolders)} {subfolder}")
                per_trial_json_data.setdefault(subfolder, {})
                path_2 = os.path.join(path_1, subfolder)
                data = [file for file in os.listdir(path_2) if file.endswith("_hypercube.pkl.gz")]
                for data_instance in data:
                    data_instance_key = data_instance[:-17]
                    path_3 = os.path.join(path_2, data_instance)
                    with open(path_3, "rb") as f:
                        pickle_instance = pickle.load(f)
                    top_win_vocab_count = pickle_instance['topic_window_word']
                    if not stored_idx2vocab:
                        idx2vocab_dir = f"{args.output_dir}/{args.idx2vocab_subdir}/{trial}_idx2vocab.json"
                        with open(idx2vocab_dir, "w") as f:
                            json.dump(pickle_instance['index_to_word'], f, indent=4, ensure_ascii=False)
                        per_trial_json_data["idx2vocab"] = idx2vocab_dir
                        stored_idx2vocab = True
                    num_top, num_win, num_vocab = top_win_vocab_count.shape
                    topic_arr = np.zeros((num_top, num_win, args.top_n))
                    top_win_vocab_dist = top_win_vocab_count / top_win_vocab_count.sum(axis=2, keepdims=True)
                    for top_id in range(num_top):
                        for win_id in range(num_win): 
                            vocab_dist = top_win_vocab_dist[top_id, win_id, :]
                            top_n_vocab = vocab_dist
                            topic_arr[top_id, win_id] = np.argsort(vocab_dist)[::-1][:args.top_n]
                    numpy_output_path = f"{args.output_dir}/{args.numpy_subdir}/{trial}_{subfolder}_{data_instance[:-17]}.npy"
                    np.save(numpy_output_path, topic_arr)
                    per_trial_json_data[subfolder][data_instance_key] = numpy_output_path

            with open(f"{args.output_dir}/{args.topdata_subdir}/{trial}_topic_data.json", "w") as f:
                json.dump(per_trial_json_data, f, indent=4, ensure_ascii=False)

        else:
            logger.info(f"trial {trial} does not have xDETM")