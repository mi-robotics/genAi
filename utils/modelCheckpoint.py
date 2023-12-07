
import torch 
import re

class Checkpoint:
    """
    Used for saving the model and relevant state dependant parts
    - this allows us to resotor the model and continue training in the future
    """

    def __init__(self, config, trainees):

        self.trainees = trainees
        self.chkpt_path = config['chkpt_path']
        
        return
    
    def load_checkpoint(self, map_location):
        chkpt = torch.load(self.chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.startswith("module."):
                        _chkpt[k.split(".", maxsplit=1)[1]] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", self.chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()
    
