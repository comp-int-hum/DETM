import os, argparse, json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="coherence_sample_50_top_50/gensim_coherence_stats")
    parser.add_argument("-t", "--trial_num", type=int, default=0)
    parser.add_argument("-o", "--output_dir", type=str, default="images/coherence_sample_50_top_50")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    trials = os.listdir(args.input_dir)
    trial_instance = trials[args.trial_num]
    trial_name = trial_instance.split('_')[0]
    print(f"navigating trial #{args.trial_num + 1}/{len(trials)}: {trial_name}")

    with open(os.path.join(args.input_dir, trial_instance), "r") as f:
        trial_data = json.load(f)

    for subfolder_idx, (subfolder_name, subfolder_data) in enumerate(trial_data.items()):
        subfolder_means = [[] for _ in range(3)]
        subfolder_labels = []
        subfolder_stds = [[] for _ in range(3)]
        # add sorting algorithms to include graph presentation
        if subfolder_name == 'DELTAS':
            subfolder_data = dict(sorted(subfolder_data.items(), key=lambda item: float(item[0].split(' ')[0])))
        elif subfolder_name in {'TOPIC_COUNT', 'VOCABULARY_SIZE', 'WINDOW_COUNT', 'WINDOW_SIZE'}:
            subfolder_data = dict(sorted(subfolder_data.items(), key = lambda item: int(item[0])))
        for config_name, config_data in subfolder_data.items():
            subfolder_labels.append(config_name)
            for metric_idx, metric_name in enumerate(['c_v', 'c_uci', 'c_npmi']): 
                subfolder_means[metric_idx].append(config_data[metric_name][0])
                subfolder_stds[metric_idx].append(config_data[metric_name][1])

        fig = plt.figure()
        x = np.arange(len(subfolder_means[0]))
        x_left = [num - 0.2 for num in x]
        x_right = [num + 0.2 for num in x]
        plt.bar(x_left, subfolder_means[0], yerr=subfolder_stds[0], width=0.2, label='c_v')
        plt.bar(x, subfolder_means[1], yerr=subfolder_stds[1], width=0.2, label='c_uci')
        plt.bar(x_right, subfolder_means[2], yerr=subfolder_stds[2], width=0.2, label='c_npmi')
        plt.xticks(x, subfolder_labels)
        plt.legend()
        plt.title(f"{trial_name}_{subfolder_name}_coherence_comparison.png")
        plt.savefig(f"{args.output_dir}/{trial_name}_{subfolder_name}_coherence_comparison.png")

