import torch
import wandb
import os

def denorm_visualizer(normed_image:torch.Tensor, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])):
    return ((normed_image.detach().permute(1, 2, 0).cpu() * std + mean)).numpy()

class WandbLogger(object):
    """ usage warning:this class only support scalar(_Tensor)
    create before training
    update(head, step, k=v) 
    flush() to force writing
    """
    def __init__(self, args):
        os.environ["WANDB_API_KEY"] = "d32c431927c09a7bf392135ee5b63acaa90f5bee"
        if args.debug:
            os.environ["WANDB_MODE"] = "dryrun"
        self.is_master = (args.rank == 0)
        if self.is_master:
            wandb.init(project='cvq_vae_debug')
            wandb.config.update(args)
        self.class_labels = {
            0: 'background', 
            1: 'aeroplane', 
            2: 'bicycle', 
            3: 'bird', 
            4: 'boat', 
            5: 'bottle', 
            6: 'bus', 
            7: 'car', 
            8: 'cat', 
            9: 'chair', 
            10: 'cow', 
            11: 'diningtable', 
            12: 'dog', 
            13: 'horse', 
            14: 'motorbike', 
            15: 'person', 
            16: 'pottedplant', 
            17: 'sheep', 
            18: 'sofa', 
            19: 'train', 
            20: 'tvmonitor', 
            255: 'void'
        }

    def update(self, **kwargs):
        if not self.is_master:
            return
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, tuple):
                wandb.log({k: wandb.Image(denorm_visualizer(v[0]), 
                                    masks={
                                        f"mask_{i}": {"mask_data":v[i], "class_labels": self.class_labels}
                                    for i in range(1, len(v))})
                            })
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            wandb.log({k:v})