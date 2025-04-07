import statistics, re, os, argparse, json, statistics 
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="coherence_sample_50_top_50/gensim_coherence_results")
    parser.add_argument("-o", "--output_dir", type=str, default="coherence_sample_50_top_50/gensim_coherence_stats")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    data = os.listdir(args.input_dir)
    
    for data_instance in data:
        with open(os.path.join(args.input_dir, data_instance), "r") as f:
            trial_data = json.load(f)

        trial_stats = {}
        for subfolder_idx, (subfolder_name, subfolder_data) in enumerate(trial_data.items()):
            print(f"navigating subfolder #{subfolder_idx}/{len(trial_data)} {subfolder_name}")
            trial_stats.setdefault(subfolder_name, {})
            for config_idx, (config_name, config_path) in enumerate(subfolder_data.items()):
                trial_stats[subfolder_name].setdefault(config_name, {})
                config_path = "coherence_sample_50_top_50" + config_path[4:]
                print(f"navigating config #{config_idx}/{len(subfolder_data)} {config_name}")
                stats_arr = np.load(config_path)
                # the three metrics are respectively c_v, c_uci, c_npmi
                mean_data = np.nanmean(stats_arr, axis=0)
                assert mean_data.shape == (stats_arr.shape[1], )
                std_data = np.nanstd(stats_arr, axis=0, ddof=1)
                assert mean_data.shape == std_data.shape

                for metric_idx, metric_name in enumerate(["c_v", "c_uci", "c_npmi"]):
                    trial_stats[subfolder_name][config_name][metric_name] = [mean_data[metric_idx].item(), std_data[metric_idx].item()]

        with open(f"{args.output_dir}/{data_instance}", "w") as f:
            json.dump(trial_stats, f, indent=4)
