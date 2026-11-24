"""Inspect negative temporal-consistency sequences without model inference.

Use the RoVF interpreter (h5py/OpenCV are already installed)::

    experiments/reference/rovf/.venv/Scripts/python.exe scripts/audit_temporal_consistency.py

Writes a new audit directory; never modifies the source experiment or masks.
Pixel changes are crop-coordinate, mask-dependent diagnostics, not optical flow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import uuid
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from PIL import Image


def run(source: Path, output: Path, root: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    sequences = pd.read_csv(source / "sequence_deltas.csv")
    raw = sequences[(sequences.threshold == 0.9) & (sequences["mode"] == "raw")].copy()
    negative = sorted(raw.loc[raw.delta < 0, "clip"].tolist())
    assert len(negative) == len(set(negative))
    manifest = pd.read_csv(source / "manifest.csv")
    measurements = pd.read_csv(source / "pair_region_metrics.csv")
    diagnostics = pd.read_csv(source / "pair_diagnostics.csv")
    primary = measurements[(measurements.threshold == 0.9) & (measurements["mode"] == "raw")]
    wide = primary.pivot(index=["identity", "clip", "pair"], columns="region", values="consistency").reset_index()
    wide["delta"] = wide.target - wide.non_target
    rng = np.random.default_rng(20261124)
    controls = []
    for identity, group in raw[raw.delta > 0].groupby("identity", sort=True):
        controls.append(str(rng.choice(sorted(group["clip"].tolist()))))
    missing = sorted(measurements.loc[measurements.consistency.isna(), "clip"].unique())
    selected = sorted(set(negative + controls + missing))
    roles = {key: ";".join(role for role, keys in [("negative",negative),("positive_control",controls),("missing_region",missing)] if key in keys) for key in selected}
    base = root / "experiments/reference/rovf"
    mask_path = base / "segmentation/results/PB_masks_Test.pkl"
    h5_path = base / "Dataset/polarbears_h5files/clips/Test/PolarBears_test.h5"
    # Existing trusted local pickle; do not load arbitrary external pickles.
    with mask_path.open("rb") as handle:
        masks = pickle.load(handle)
    pair_pixels, audit_frames, frame_selection = [], [], []
    with h5py.File(h5_path, "r") as handle:
        for key in selected:
            clip_manifest = manifest[manifest["clip"] == key].sort_values("stored_index")
            indices = clip_manifest.stored_index.to_numpy(dtype=int)
            images = np.asarray(handle[key])[indices]
            clip_masks = np.asarray(masks[key])[indices].astype(bool)
            assert images.shape == (8,224,224,3) and clip_masks.shape == (8,224,224)
            identity = int(clip_manifest.identity.iloc[0])
            worst = wide[wide["clip"] == key].dropna(subset=["delta"]).sort_values("delta").iloc[0]
            worst_pair = int(worst["pair"])
            missing_pairs = measurements.loc[(measurements["clip"]==key) & measurements.consistency.isna(),"pair"].unique()
            sample = sorted({0,4,7,worst_pair,worst_pair+1} | {t for pair in missing_pairs for t in [int(pair),int(pair)+1]})
            for t in range(8):
                raw_path = root / clip_manifest.raw_path.iloc[t]
                with Image.open(raw_path) as original:
                    raw_size = original.size
                audit_frames.append(dict(clip=key,identity=identity,role=roles[key],t=t,
                    stored_index=int(indices[t]),source_frame=int(clip_manifest.source_frame.iloc[t]),
                    raw_path=clip_manifest.raw_path.iloc[t],raw_width=raw_size[0],raw_height=raw_size[1],
                    mask_area=float(clip_masks[t].mean()),qc_selected=t in sample,
                    visual_status="pending",independent_gt="Not available"))
                if t in sample:
                    frame_selection.append(dict(clip=key,t=t,rule="first/middle/last plus lowest-delta and missing-region pair endpoints",role=roles[key]))
            for t in range(7):
                difference = np.abs(images[t+1].astype(float)-images[t].astype(float)).mean(-1)/255
                target = clip_masks[t] & clip_masks[t+1]
                non_target = ~clip_masks[t] & ~clip_masks[t+1]
                pair_pixels.append(dict(clip=key,identity=identity,pair=t,role=roles[key],
                    target_rgb_mad=float(difference[target].mean()) if target.any() else np.nan,
                    non_target_rgb_mad=float(difference[non_target].mean()) if non_target.any() else np.nan,
                    target_pixels=int(target.sum()),non_target_pixels=int(non_target.sum())))
            fig, axes = plt.subplots(2,8,figsize=(16,4.7))
            for t in range(8):
                rgb = images[t].astype(float)/255
                overlay=rgb.copy()
                overlay[clip_masks[t]]=overlay[clip_masks[t]]*.65+np.array([.35,0,0])
                axes[0,t].imshow(rgb)
                axes[1,t].imshow(overlay)
                axes[0,t].set_title(f't={t}, stored={indices[t]}\nF{clip_manifest.source_frame.iloc[t]}',fontsize=8)
                axes[1,t].set_title('QC sample' if t in sample else '',fontsize=8)
                for ax in axes[:,t]:
                    ax.axis('off')
            delta=float(raw.loc[raw["clip"]==key,"delta"].iloc[0])
            fig.suptitle(f'{key} | {roles[key]} | raw delta={delta:+.4f} | top RGB / bottom generated mask overlay',fontsize=11)
            fig.tight_layout()
            fig.savefig(output/f'{key}_all_frames.png',dpi=120)
            plt.close(fig)
            print(f'Audited inputs: {key}',flush=True)
    # Fixed, purposive coordinates from the reviewed run. A fresh audit leaves
    # visual labels pending: these probes are not independent segmentation GT.
    fixed_probes=[('002_104',0,160,80),('002_087',0,90,55),('011_084',10,65,65),
                  ('001_096',8,60,100),('002_107',0,175,90),('002_087',0,90,40),('011_084',10,90,65)]
    available_probes=[probe for probe in fixed_probes if probe[0] in selected]
    probe_rows=[]
    if available_probes:
        fig,axes=plt.subplots(len(available_probes),2,figsize=(8,4*len(available_probes)),squeeze=False)
        with h5py.File(h5_path,'r') as handle:
            for i,(key,stored,x,y) in enumerate(available_probes):
                mask=np.asarray(masks[key][stored]).astype(bool)
                py,px=y//14,x//14
                probe_rows.append(dict(clip=key,stored_index=stored,x=x,y=y,
                    generated_mask_target=bool(mask[y,x]),patch_y=py,patch_x=px,
                    patch_occupancy=float(mask[py*14:(py+1)*14,px*14:(px+1)*14].mean()),
                    reviewer='pending; not independent human GT',visual_label='pending',
                    selection_bias='Fixed post-hoc probes from reviewed run; not random'))
                axes[i,0].imshow(np.asarray(handle[key][stored]))
                axes[i,1].imshow(mask,cmap='gray',vmin=0,vmax=1)
                for ax in axes[i]:
                    ax.scatter([x],[y],marker='+',c='lime',s=140,linewidths=2)
                    ax.set_xticks(np.arange(0,225,28));ax.set_yticks(np.arange(0,225,28));ax.grid(alpha=.25)
                axes[i,0].set_title(f'{key} stored={stored}, probe ({x},{y})')
                axes[i,1].set_title(f'mask at probe: {int(mask[y,x])}')
        fig.tight_layout();fig.savefig(output/'mask_pixel_probes.png',dpi=110);plt.close(fig)
    pd.DataFrame(probe_rows,columns=['clip','stored_index','x','y','generated_mask_target','patch_y','patch_x',
                                   'patch_occupancy','reviewer','visual_label','selection_bias']).to_csv(output/'mask_pixel_probes.csv',index=False)
    del masks
    pixels=pd.DataFrame(pair_pixels)
    pixels.to_csv(output/'pair_pixel_diagnostics.csv',index=False)
    pd.DataFrame(audit_frames).to_csv(output/'frame_qc.csv',index=False)
    pd.DataFrame(frame_selection).to_csv(output/'qc_selection.csv',index=False)
    features=wide.dropna(subset=['target','non_target']).groupby(['identity','clip'])[['target','non_target']].mean().reset_index()
    metrics=diagnostics.groupby('clip').agg(motion=('motion_per_source_frame','mean'),shape_change=('shape_change','mean')).reset_index()
    summary=raw.merge(features,on=['identity','clip']).merge(metrics,on='clip')
    for column in ['motion','shape_change','target','non_target']:
        summary[column+'_percentile_among_62']=summary[column].rank(pct=True)*100
    summary['negative']=summary.delta<0
    summary['role']=summary['clip'].map(roles).fillna('not_visually_sampled')
    summary.to_csv(output/'all_sequence_diagnostics.csv',index=False)
    summary.groupby('negative')[['delta','motion','shape_change','target','non_target']].median().to_csv(output/'negative_positive_group_medians.csv')
    selected_summary=summary[summary['clip'].isin(selected)].merge(pixels.groupby('clip')[['target_rgb_mad','non_target_rgb_mad']].mean().reset_index(),on='clip')
    selected_summary['visual_motion']='Not verified'
    selected_summary['visual_background']='Not verified'
    selected_summary['visual_mask']='Not verified'
    selected_summary['causal_attribution']='Not identifiable from generated masks and crops alone'
    selected_summary.to_csv(output/'sequence_review.csv',index=False)
    archive=root/'data/raw/PolarBearVidID.zip'
    extensions={}
    if archive.is_file():
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if not name.endswith('/'):
                    suffix=Path(name).suffix.lower()
                    extensions[suffix]=extensions.get(suffix,0)+1
    video_suffixes={'.mp4','.avi','.mov','.mkv','.webm','.mpeg','.mpg','.mts'}
    # Filter names before filesystem stats: the release contains 138k JPGs.
    inventory=[(Path(folder)/name).relative_to(root).as_posix()
               for folder,_,names in os.walk(root/'data')
               for name in names if Path(name).suffix.lower() in video_suffixes]
    provenance=dict(source=source.relative_to(root).as_posix(),negative_clips=negative,
        positive_controls=controls,missing_region_clips=missing,selected_clips=selected,
        negative_count=len(negative),reviewed_count=0,archive_extensions=extensions,
        local_video_files_under_data=inventory,
        sources={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['manifest.csv','sequence_deltas.csv','pair_region_metrics.csv','pair_diagnostics.csv']},
        interpretation='Crop-coordinate, generated-mask-conditioned diagnostics; no GT mask IoU, physical motion or causal attribution',
        review_status='visual review pending')
    (output/'audit_metadata.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    availability=dict(local_uncropped_video='Not available under data/' if not inventory else inventory,
        archive_extensions=extensions,official_sources=['https://zenodo.org/records/7564529','https://www.mdpi.com/2076-2615/13/5/801'],
        public_uncropped_download='Not verified by this offline script; consult the dated audit report',
        decision='Independent target mask review first; full-background/occluder hypothesis requires uncropped videos and separate labels',
        communication='No download or author contact initiated')
    (output/'raw_video_availability.json').write_text(json.dumps(availability,indent=2),encoding='utf-8')
    print('Saved:',output)
    return provenance


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='experiments/polarbear/temporal_consistency_identity_test/20261124T120950Z-d2ad62')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=(root/args.source).resolve()
    output=source/'audits'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    run(source,output,root)


if __name__=='__main__':
    main()
