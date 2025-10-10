
import argparse, yaml
from stormfusion.training.loop import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/base.yaml')
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--outdir', type=str, default='outputs')
    p.add_argument('--data.tiny', dest='data_tiny', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_tiny:
        cfg['data']['tiny'] = True
    run(cfg, max_steps=args.max_steps, outdir=args.outdir)

if __name__ == '__main__':
    main()
