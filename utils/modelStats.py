import torch
import wandb
import os

from utils.misc import flatten_dict

os.environ['WANDB_SILENT'] = 'true'


class RunningStatistics:
    """
    Keeping track of values over an epoch with am interface to save values to wandb
    """


    def __init__(self, config, **kwargs):
        """
        @param: kwargs - values to track over an epoch
        @param: use_wandb
        @param: config - experiments config in full
        @param: project_name
        @param: experiment_name
        """

        self.use_wandb = config['use_wandb']
        self.config = config
        self._init_wandb()

        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)
    
    def _init_wandb(self):
        if self.use_wandb:
            run = wandb.init(
                project=self.config['project_name'],
                name=self.config["experiment_name"],
                config=flatten_dict(self.config)
            )
        return
    
    def wand_log(self):
        if self.use_wandb:
            wandb.log(self.extract())
        return 
    
    def wand_end(self):
        if self.use_wandb:
            wandb.finish()

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v / self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)
    








if __name__ == '__main__':

    track = {
        "loss":None,
        "elbo":None
    }

    stats = RunningStatistics(use_wandb=True, 
                              config={
                                  "use_wandb":True,
                                  'project_name':"demo_setup",
                                  "experiment_name":"experiment-unet3",
                                  "learning-rate":1e-3,
                                  "unet":{
                                      "d":1,
                                      'arr':[100,20]
                                  }
                                },
                              **track)
    

    stats.update(2, **{"loss":10, 'elbo':5})
    stats.wand_log()
    stats.update(2, **{"loss":7, 'elbo':9})
    stats.wand_log()
    stats.update(2, **{"loss":5, 'elbo':10})
    stats.wand_log()
  
    wandb.finish()

