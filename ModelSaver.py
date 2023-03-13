from pathlib import Path

from timm.utils import get_state_dict

from Distributed import save_on_master

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    """main usage
    requiring the args.output_dir
    default save epoch, model/ddp
    optionally scalar and args and optimizer
    optionally model_ema
    custom changes can be made to the to_save dict and add **kwargs
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)