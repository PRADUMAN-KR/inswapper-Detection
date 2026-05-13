# Dataset Manifests

Keep raw images outside git. The API and training code expect CSV manifests with:

```csv
path,label,source
data/raw/real/0001.jpg,0,celebdf
data/raw/fake/0001.jpg,1,inswapper
```

Use `label=0` for real images and `label=1` for fake or swapped images.

Final training manifests may also include:

```csv
path,label,fake_type,is_inswapper,is_gan,boundary_label,quality_label,source,video_id,identity_id
data/raw/processed_crops/0001_tight.jpg,1,inswapper,1,0,1,0,inswapper,video_001,person_001
```

`quality_label` is an integer class for compression/quality buckets, for example `0=clean`, `1=compressed`, `2=low_quality`.

Recommended dataset mix:

- Real faces from Celeb-DF, FaceForensics++, DFDC, or your owned/licensed data.
- Fake faces generated with INSwapper and other GAN/diffusion face-swap pipelines.
- Split by identity and source video, not by frame, to avoid leakage.
