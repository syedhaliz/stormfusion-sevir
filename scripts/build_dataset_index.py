
import argparse, pandas as pd, os, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=str, required=True, help='SEVIR catalog CSV path')
    p.add_argument('--out', type=str, default='data/index.json')
    args = p.parse_args()
    if not os.path.exists(args.catalog):
        print('Catalog not found; create a placeholder or download it.')
        return
    df = pd.read_csv(args.catalog)
    # TODO: filter and build per-task indices
    json.dump({'todo': 'build mapping from event id to HDF5 path, key, t0'}, open(args.out,'w'))
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
