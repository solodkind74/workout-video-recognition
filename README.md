# vidoemodel: recognize gym exercises from video

PyTorch on the M5 GPU (MPS), $0. Dataset: [Workout/Exercises Video](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
(Kaggle, 652 YouTube clips, 22 exercises, ~5 s each, 12 to 62 clips per class). Three models compared:

1. **Frozen image model (DINOv2-small) per frame + small head**: mean of frames, or a 2-layer temporal transformer
2. **Frozen video model (VideoMAE-base, Kinetics-400 fine-tuned) + linear head**
3. **VideoMAE with its top 4 of 12 blocks fine-tuned** (28M trainable params) + video augmentation

```sh
uv sync
uv run python -c "import kagglehub; kagglehub.dataset_download('hasyimabdillah/workoutfitness-video')"  # 4.3 GB, set KAGGLEHUB_CACHE=data/kaggle
uv run python prep.py              # decode at 8 fps, 224x224 center crop -> data/frames  (~2 min, 4.2 GB)
uv run python features.py          # DINOv2 per-frame features                            (~3.5 min)
uv run python split.py             # group near-duplicate clips, write group + random splits
uv run python train_feats.py --head temporal --out runs/dino_temporal_group   # model 1, seconds
uv run python features_video.py    # VideoMAE features per 16-frame window, + time-shuffled copy (~15 min)
uv run python train_vmae.py --out runs/vmae_linear_group                      # model 2, seconds
uv run python finetune.py --out runs/finetune                                  # model 3, ~36 min
uv run python cam.py                         # live webcam, top-3 overlay (q quits)
uv run python cam.py --test squat -n 5       # record + score YOUR clips -> data/mine/ (eval only)
uv run python cam.py --score                 # re-score all saved clips
```

## Results (test split grouped by source video, 108 clips; mean of 3 seeds where shown)

| | Test acc | Macro-F1 | Acc, frames shuffled in time | Train time |
|---|---|---|---|---|
| DINOv2 frames, mean head (3 seeds) | 53% | 0.49 | 53% | seconds |
| DINOv2 frames, temporal transformer (3 seeds) | 53% | 0.47 | 53% | seconds |
| **VideoMAE frozen + linear (3 seeds)** | **88%** | **0.86** | 82% | seconds (+15 min features) |
| VideoMAE frozen + MLP (3 seeds) | 86% | 0.83 | 80% | seconds |
| VideoMAE top-4 blocks fine-tuned (1 seed) | 89% | 0.84 | 82% | 36 min |
| *Same DINOv2 / VideoMAE-linear on a random split* | *88% / 95%* | *0.84 / 0.95* | | |

### Lesson 1: the split matters more than the model
Many clips are cut from the same YouTube video (same person, same gym). Nearest-neighbor cosine on mean
DINOv2 embeddings: 318/652 clips have a twin at >0.95. `split.py` groups them (single linkage within a class,
cosine > 0.8, eyeballed: 0.85 pairs = same person and gym, 0.75 pairs = different people) and assigns whole groups
to train/val/test. In a random split **81 of 98 test clips had a same-source clip in train**. The DINOv2 model
scores 88% there and 53% on the grouped split. On the random split it recognizes the gym and the person, not the exercise.

### Lesson 2: image features don't see motion; video features do
Shuffling the frame order at test time does not change DINOv2 predictions at all, even with a temporal
transformer. With 450 training clips it never learned to use order. VideoMAE was pretrained on video, so shuffling
costs it 6 points (88% → 82%). On the random split shuffling doesn't matter even for VideoMAE (95% → 95%),
another sign that random-split scores come from scenery.

### Lesson 3: the pretrained model's existing labels explain most of the score
VideoMAE already has Kinetics-400 classes for several exercises (`tools/analyze.py`). Squat, push-up, deadlift,
pull up and bench press map to their Kinetics label in 98–100% of clips. All curls, raises, pushdowns and lat
pulldowns fall under "front raises", and the linear head learns to separate them. The remaining errors are
close variants: hammer vs barbell curl, flat vs incline bench, Romanian vs regular deadlift.

### Lesson 4: fine-tuning didn't pay off at this data size
36 minutes of fine-tuning (vs seconds for the linear head) gave the same accuracy and a lower macro-F1. With ~450
clips from a few dozen sources per class at most, the frozen features are already as good as the data allows. Note: epochs 4–6 ran ~4x
slower because the 16 GB Mac started swapping.

## Ideas to try next
- **Test on yourself** (`cam.py --test`): bodyweight squat, push-up, plank, russian twist, leg raises, tricep
  dips (on a chair). Gym footage vs a home webcam is the same sim-to-real question as the `jarvis` wake word.
- **Rep counting**: VideoMAE/DINOv2 similarity between frames over time is periodic. Count peaks
  (or use a pose model like MediaPipe and count knee-angle cycles).
- **More data per class**: bigger fitness sets (e.g. Fitness-AQA, or filter Kinetics-700 exercise classes).
- **Smaller/faster**: distill into a tiny model, or use VideoMAE-small; export to CoreML for a phone.
- **Video generation** (next level): train a tiny video diffusion model on Moving MNIST from scratch.
