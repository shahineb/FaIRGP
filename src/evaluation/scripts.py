import os
import torch


def dump_state_dict(model, output_dir):
    dump_path = os.path.join(output_dir, 'state_dict.pt')
    torch.save(model.cpu().state_dict(), dump_path)
