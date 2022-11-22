import os
import json
import torch
import matplotlib.pyplot as plt
from src.evaluation import visualisation as vis


def dump_state_dict(model, output_dir):
    dump_path = os.path.join(output_dir, 'state_dict.pt')
    torch.save(model.cpu().state_dict(), dump_path)


def dump_logs(logs, output_dir):
    dump_path = os.path.join(output_dir, 'logs.json')
    with open(dump_path, 'w') as f:
        json.dump(logs, f)


def dump_plots(posterior_dist, test_scenarios, model, output_dir):
    dump_path = os.path.join(output_dir, 'prediction.jpg')
    _ = vis.plot_scenario_prediction(posterior_dist=posterior_dist,
                                     test_scenarios=test_scenarios,
                                     model=model)
    plt.savefig(dump_path)
    plt.close()
