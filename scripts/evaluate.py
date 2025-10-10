
import argparse, yaml, torch
from stormfusion.training.loop import build_loaders, evaluate, make_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/base.yaml')
    p.add_argument('--data.tiny', dest='data_tiny', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_tiny:
        cfg['data']['tiny'] = True
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['device']=='cuda' else 'cpu')
    model = make_model(cfg).to(device)
    _, dl_va = build_loaders(cfg)
    loss, s = evaluate(model, dl_va, device)
    print({'loss': float(loss), 'CSI@74': s[74]['CSI']})

if __name__ == '__main__':
    main()
