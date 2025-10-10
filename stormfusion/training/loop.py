
import os, yaml, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stormfusion.models.unet2d import UNet2D
from stormfusion.training.metrics import mse
from stormfusion.training.forecast_metrics import scores
from stormfusion.data.sevir_dataset import SevirNowcastDataset, build_tiny_index

def make_model(cfg):
    if cfg['model']['name'] == 'unet2d':
        return UNet2D(cfg['model']['in_channels'], cfg['model']['out_channels'],
                      base_ch=cfg['model'].get('base_channels',32))
    else:
        raise NotImplementedError(cfg['model']['name'])

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = mse(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if math.isfinite(loss.item()):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()*x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = 0.0
    agg_scores = None
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        tot += mse(pred,y).item()*x.size(0)
        s = scores(pred, y)
        if agg_scores is None:
            agg_scores = {k:{m:0.0 for m in s[k]} for k in s}
        for k in s:
            for m,v in s[k].items(): agg_scores[k][m] += v * x.size(0)
    N = len(loader.dataset)
    for k in agg_scores:
        for m in agg_scores[k]:
            agg_scores[k][m] /= N
    return tot/N, agg_scores

def build_loaders(cfg):
    tiny = cfg['data'].get('tiny', False)
    root = cfg['data']['root']
    catalog_path = cfg['data']['catalog_path']

    if tiny:
        train_index = build_tiny_index(
            catalog_path=catalog_path,
            ids_txt="data/samples/tiny_train_ids.txt",
            sevir_root=root,
            modality=cfg['data'].get('modality', 'vil')
        )
        val_index = build_tiny_index(
            catalog_path=catalog_path,
            ids_txt="data/samples/tiny_val_ids.txt",
            sevir_root=root,
            modality=cfg['data'].get('modality', 'vil')
        )
    else:
        # TODO: parse real catalog
        train_index = build_tiny_index(
            catalog_path=catalog_path,
            ids_txt="data/samples/tiny_train_ids.txt",
            sevir_root=root,
            modality=cfg['data'].get('modality', 'vil')
        )
        val_index = build_tiny_index(
            catalog_path=catalog_path,
            ids_txt="data/samples/tiny_val_ids.txt",
            sevir_root=root,
            modality=cfg['data'].get('modality', 'vil')
        )

    ds_tr = SevirNowcastDataset(
        train_index,
        input_steps=cfg['data'].get('input_steps', 12),
        output_steps=cfg['data'].get('output_steps', 1)
    )
    ds_va = SevirNowcastDataset(
        val_index,
        input_steps=cfg['data'].get('input_steps', 12),
        output_steps=cfg['data'].get('output_steps', 1)
    )

    dl_tr = DataLoader(ds_tr, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    dl_va = DataLoader(ds_va, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    return dl_tr, dl_va

def run(cfg, max_steps=None, outdir="outputs"):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['device']=='cuda' else 'cpu')
    model = make_model(cfg).to(device)
    dl_tr, dl_va = build_loaders(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['optim']['lr'])
    steps = 0
    while steps < (max_steps or cfg['train']['max_steps']):
        train_one_epoch(model, dl_tr, opt, device)
        steps += len(dl_tr)
        val_loss, val_scores = evaluate(model, dl_va, device)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'val_scores.yaml'), 'w') as f:
            yaml.safe_dump({'loss': float(val_loss), 'scores': val_scores}, f)
        print(f"VAL loss={val_loss:.4f} CSI@74={val_scores[74]['CSI']:.3f}")
    return model
