# Elastic toy training data

50 elastic samples from `_pbd_soft_air_cpu_full`: 10 each with 1–5 objects.
Selected as the first 10 training entries per object count in the full dataset split:
`JmT4D/configs/splits/12345_obj_10000total_9500train_250val_250test_exact_per_object_soft_air_cpu_full.json`
(9,500 train, 250 validation, 250 test). No validation/test samples are included.

Each `<N>_obj/sample_<0–9>/` contains `sample.npz` and `trajectory.mp4`.
`sample_mapping.csv` maps toy directories to original sample IDs and selectors.
Files are exact copies; embedded original NPZ identifiers are unchanged.

Precomputed source: `JmT4D/_soft_material_full_mix/precomputed_v356_maxobj10/soft_air_cpu_full`.
Video source: `JmT4D/_pbd_soft_air_cpu_full/<N>_obj/sample_<ID>/trajectory.mp4`.
The inspected NPZ has vertex trajectories shaped `(49, 356, 3)`, masks,
object IDs, and initial velocities. Mesh topology is not included.

For elastic-only loading, use `data_toy/elastic` as the precomputed root and
`data_toy/elastic/split.json` as the split file. `sample_0` from each object count is validation (5 samples); `sample_1` through
`sample_9` are training (45 samples). `eval` aliases `val`; `test` is empty.
Use elastic material conditioning (`[1.0]`) when building a standalone pipeline.
For combined rigid/elastic training, use the [toy launcher](../README.md#training).
