import os, json, argparse, logging, gzip, random
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import numpy as np

METRICS = ['c_v', 'c_uci', 'c_npmi']

class PerTrialDictionary:

    def __init__(self, idx2token):
        self.id2token = {int(k): v for k, v in idx2token.items()}
        self.token2id = {v: int(k) for k, v in idx2token.items()}

    def __len__(self):
        return len(self.token2id)

    def __contains__(self, item):
        # Check if the item is either a token or an ID
        return item in self.id2token

logger = logging.getLogger("generate_gensim_coherence")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    filename="logger.out", level=logging.DEBUG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="data/topic_data")
    parser.add_argument("-t", "--texts_dir", type=str, default="/data/tlippin1/tlippin1/detm-shootout/work")
    parser.add_argument("-o", "--output_dir", type=str, default="data")
    parser.add_argument("-c", "--coherence_subdir", type=str, default="gensim_coherence_results")
    parser.add_argument("-ca", "--conumpy_subdir", type=str, default="coherence_numpy_arr")
    parser.add_argument("-n", "--top_n", type=int, default=20)
    parser.add_argument("-s", "--sample_per_trial", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.coherence_subdir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.conumpy_subdir}", exist_ok=True)

    trials = os.listdir(args.input_dir)
    for trial_idx, trial in enumerate(trials):
        trial_name = trial.split("_topic_data")[0]
        logger.debug(f"navigating #{trial_idx + 1}/{len(trials)} trial {trial_name}")
        per_trial_coherence_data = {}

        with open(os.path.join(args.input_dir, trial), "r") as f:
            json_stats = json.load(f)
        
        if not json_stats.get('idx2vocab', None):
            continue
        
        with open(json_stats['idx2vocab'], 'r') as f:
            idx2token = json.load(f)
        trial_dictionary = PerTrialDictionary(idx2token)

        trial_texts = []
        try:
            with gzip.open(f"{args.texts_dir}/{trial_name}/test_data.jsonl.gz", "rb") as f:
                for line in tqdm(f):
                    j = json.loads(line)
                    trial_texts.extend(j.get('content', []))
        except Exception as e:
            logger.debug(f"get exception : {str(e)}")
            pass

        logger.debug(f"has a total of {len(trial_texts)} instances of text as corpus data for trial {trial_name}")

        for subfolder_idx, (subfolder_name, subfolder_data) in enumerate(json_stats.items()):

            logger.debug(f"navigating #{subfolder_idx + 1}/{len(json_stats)} subfolder <{subfolder_name}>")
            if subfolder_name == "idx2vocab":
                continue

            per_trial_coherence_data.setdefault(subfolder_name, {})

            for config_idx, (config_name, config_data) in enumerate(subfolder_data.items()):
                logger.debug(f"navigating #{config_idx + 1}/{len(subfolder_data)} config <{config_name}>")
               
                top_win_data = np.load(config_data)
                if args.top_n:
                    top_win_data = top_win_data[:, :, :args.top_n]
                num_top, num_win, num_vocab = top_win_data.shape
                logger.debug(f"trial {trial_name} subfolder {subfolder_name} config {config_name} has {num_win} windows and {num_top} topics")
                per_config_arr = np.zeros((num_top, num_win, len(METRICS))) if not args.sample_per_trial else np.zeros((args.sample_per_trial, len(METRICS)))
                topics = top_win_data.reshape((num_top * num_win, num_vocab))
                if args.sample_per_trial and num_top * num_win > args.sample_per_trial:
                    sampled_rows = random.sample(list(topics), k=args.sample_per_trial)  # k = number of samples
                    topics = np.array(sampled_rows)
                for metric_idx, metrics in enumerate(METRICS):

                    cm = CoherenceModel(topics=topics,
                                        dictionary=trial_dictionary, texts=trial_texts, coherence=metrics)
                    result = np.array(cm.get_coherence_per_topic())
                    # result = np.arange(topics.shape[0])
                    if not args.sample_per_trial:
                        per_config_arr[:, :, metric_idx] = result.reshape((num_top, num_win))
                    else:
                        if result.shape[0] < args.sample_per_trial:
                            num_to_pad = args.sample_per_trial - result.shape[0]
                            result = np.pad(result, (0, num_to_pad), mode='constant', constant_values=np.mean(result))

                        per_config_arr[:, metric_idx] = result
                            
                per_config_numpy_path = f"{args.output_dir}/{args.conumpy_subdir}/{trial_name}_{subfolder_name}_{config_name}_coherence.npy"
                np.save(per_config_numpy_path, per_config_arr)
                per_trial_coherence_data[subfolder_name][config_name] = per_config_numpy_path

        per_trial_data_path = f"{args.output_dir}/{args.coherence_subdir}/{trial_name}_coherence_data.json"
        with open(per_trial_data_path, "w") as f:
            json.dump(per_trial_coherence_data, f, indent=4)
