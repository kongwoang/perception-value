# Real YOLOv8s perception on the nuPlan external track

**What this is:** the Track B nuPlan result re-measured with real detections. It replaces the transported KITTI miss
profile with real YOLOv8s 320 / 640 detections on nuPlan CAM_F0 images, for PDM-Closed and IDM, on the same 1,440
states (60 scenarios, 34 logs) as Track B and the benchmark.

**Provenance:**
* Pre-registration: `RESEARCH_LOG.md`, 2026-09-14 14:44 (commit `9d96902`).
* One amendment before any branch was built or scored: the ground plane of the false-positive lift (commit `a6b494b`).

**Files:**
* Every number below comes from `results/final/nuplan_real_perception_cells.csv`, `nuplan_real_perception_checks.json`
  and `nuplan_real_perception_recall.csv`.
* Full tables, including the test split, sensitivities and components: `docs/nuplan_real_perception_tables.md`.

**Notation:**
* V = J(CHEAP) − J(FULL).
* harm rate = P(V < 0 | V ≠ 0).
* ρ = Σmax(−V, 0) / Σmax(V, 0).
* Reductions are relative to the all-CHEAP cost.
* CIs are log-level bootstrap, 1,000 draws.

## 1. What was run

| stage | what | outcome |
|---|---|---|
| **A. Data** | HTTP Range fetch of only the CAM_F0 members inside the Task 3 scenario windows | 12,921 images; CRC32 and JPEG decode verified for all; 2.77 GB of image data (cap 3.5 GB); 360 MB of ZIP directories (cap 400 MB); every shard's logs equal its metadata file group |
| **B1. Detection** | existing TensorRT engines, conf ≥ 0.10 cached | 320: 14.4 ms, 44.7 mJ per frame; 640: 23.6 ms, 105.8 mJ per frame; no throttling |
| **B2. Alignment** | nearest CAM_F0 image per state iteration | \|Δt\| ≤ 40.9 ms, none above 50 ms |
| **B3. Branches** | tracks projected into the image; Hungarian IoU matching per mode; unmatched in-camera tracks removed; unmatched detections added as false positives | per state: 18.5 eligible in-camera tracks; CHEAP keeps 5.4 and adds 0.83 FPs; FULL keeps 7.7 and adds 1.68 |
| **B4. Checks, before scoring** | 20 overlays; match IoU (median 0.66); recall against KITTI and against the transported model; the lift on true boxes | all passed or reported; see §4 |
| **C. Planners** | 82's `make_planner`, `score` and `filtered_history`, unchanged, driven by a filter with the same interface; 7 branches per planner; identical observations share one planner call | both planners: reference reproduces Track B's stored values on all 1,440 states with 0 mismatches; identity branch reproduces reference |

**Projection method (B3):**
* Lidar boxes are carried into the camera through the ego pose at image time, the extrinsic, the distortion and the
  intrinsic.
* Matching is Hungarian on IoU (≥ 0.3), class-compatible, per mode.
* An in-camera vehicle, pedestrian or bicycle track is kept only if that mode matched it.
* Unmatched detections become false positives by a flat-ground lift: train ∪ val road plane, class median size,
  zero velocity.

**Branches run (C):**

| branch | detections used | false positives |
|---|---|---|
| primary | 0.25 / 0.25, IoU 0.3 | yes |
| nofp | 0.25 / 0.25, IoU 0.3 | no |
| s1 | FULL threshold count-matched on train ∪ val (0.477) | yes |
| iou50 | 0.25 / 0.25, IoU 0.5 | yes |

## 2. Pre-registered reading: **intermediate**

Primary variant, all 1,440 states.

| aggregate | harm, PDM-Closed | harm, IDM | ρ, PDM-Closed | ρ, IDM | B-F1 fires (both < 5%) | both ≥ 20% and ρ ≥ 0.20 |
|---|---|---|---|---|---|---|
| safety | 44.4% | 16.7% | 0.33 | 0.01 | no | no |
| scalar_J | 37.5% | 48.0% | 0.35 | 0.06 | no | no |

**Why the reading is "intermediate":**
* The falsifier does not fire: harm stays at or above 16.7% for both planners on both aggregates.
* "Consistent with the benchmark" also does not hold, because of IDM:
  * on safety it responds on only 6 states and its one harmful state is tiny (ρ 0.013);
  * on scalar_J its harm rate is high, but the harm mass is small (ρ 0.06).
* PDM-Closed alone clears both bars on both aggregates.
* The IDM part of the reading was already fixed once IDM finished, before PDM-Closed completed (RESEARCH_LOG interim
  entry).

## 3. Transported profile vs real perception (all 34 logs)

| planner | aggregate | profile | affected states | V>0 / V<0 | harm rate [95% CI] | ρ [95% CI] | all-FULL reduction | oracle@20 reduction |
|---|---|---|---|---|---|---|---|---|
| PDM-Closed | safety | transported | 70 | 44 / 26 | 37.1% [24.5, 50.7] | 0.30 [0.12, 0.63] | 14.7% | 21.1% |
| PDM-Closed | safety | **real** | **45** | 25 / 20 | **44.4%** [26.5, 69.2] | **0.33** [0.02, 0.93] | 9.8% | 14.7% |
| PDM-Closed | scalar_J | transported | 175 | 122 / 53 | 30.3% [16.0, 47.6] | 0.30 [0.12, 0.61] | 12.6% | 17.9% |
| PDM-Closed | scalar_J | **real** | **104** | 65 / 39 | **37.5%** [27.3, 46.9] | **0.35** [0.07, 0.89] | 8.3% | 12.7% |
| IDM | safety | transported | 35 | 24 / 11 | 31.4% [13.8, 52.0] | 0.36 [0.13, 0.87] | 3.9% | 6.1% |
| IDM | safety | **real** | **6** | 5 / 1 | **16.7%** [0.0, 16.7] | **0.01** | 0.5% | 0.5% |
| IDM | scalar_J | transported | 96 | 57 / 39 | 40.6% [28.0, 53.6] | 0.37 [0.13, 0.86] | 2.0% | 3.2% |
| IDM | scalar_J | **real** | **25** | 13 / 12 | **48.0%** [28.6, 66.7] | **0.06** [0.01, 4.72] | 0.2% | 0.2% |

**On the 9 test logs,** PDM-Closed real safety gives:
* 27 affected states;
* harm 33.3%, ρ 0.36;
* all-FULL 25.2%, oracle@20 39.4%.

IDM on the test logs is again 6 states at 16.7%. Both test readings are intermediate.

### What the comparison shows

**1. With real perception, decision value is sparser, and the scale depends on the planner.**

| planner | aggregate | affected states, transported → real | all-FULL reduction, transported → real |
|---|---|---|---|
| PDM-Closed | safety | 70 → 45 (−36%) | 14.7% → 9.8% |
| IDM | safety | 35 → 6 (~6× fewer) | 3.9% → 0.5% |

IDM, which reacts only to agents in its path, is nearly inert to the 320 → 640 choice under real detection.

**2. The transported profile had exaggerated the fidelity gap.**
* On the same 26,611 in-camera tracks, measured recall is 0.293 at 320 and 0.415 at 640. The KITTI model predicts
  0.204 and 0.540, a gap of 0.34 against a measured 0.12.
* The gap is smallest where braking decisions live: 0.647 vs 0.648 inside 10 m.
* So both modes usually see the lead vehicle.

**3. Sign variation survives with real perception, for the planner that responds.**
* PDM-Closed's harm rate and ρ are at least as high as under the transported profile: 44% vs 37% harm on safety,
  and ρ 0.33 vs 0.30.
* The loss channel is measured, not assumed: 1.9% of tracks are hit at 320 but missed at 640 (KITTI: 1.5%).

**4. For PDM-Closed, most of the real-perception value runs through false positives.**

| PDM-Closed branch | FULL keeps / adds FPs per state | affected, safety | affected, scalar_J | harm, safety | ρ, safety |
|---|---|---|---|---|---|
| primary | 7.7 / 1.68 | 45 | 104 | 44.4% | 0.33 |
| nofp (false positives dropped) | 7.7 / 0 | 6 | 28 | 33% [0, 67] | 0.30 |
| s1 (FULL threshold count-matched) | 5.8 / 0.55 | 35 | 76 | 25.7% | 0.25 |

* Dropping false positives leaves the harm rate similar, but on very few states.
* At a shared 0.25 threshold, FULL adds twice as many false positives per state as CHEAP.
* With the count-matched threshold, value and sign variation remain.

This branch is a registered sensitivity, and the flat-ground lift that places these false positives is accurate only
within about 30 m (§4). So this is a property of the pipeline as registered, not a clean physical attribution.

**5. IoU 0.5 matching changes little for PDM-Closed:** safety harm 47.1%, ρ 0.37, 51 affected states.

## 4. Checks and their limits

| check | result |
|---|---|
| reference re-run vs Track B | 0 mismatches in collision, clearance and both log deviations, both planners, all 1,440 states |
| identity branch (all in-camera tracks matched, no FPs) | equal to reference observation on all states × 4 buffer iterations; equal planner outputs on 120 planned states |
| projection | visually aligned in the overlays inspected (3 cities, 6 vehicles); distortion monotone to the clamp radius in all 34 logs |
| false-positive lift, applied to true vehicle boxes | median \|range error\| 0.30 m within 15 m, 0.81 m at 15–30 m, 6.3 m at 30–60 m, with a long tail; lifts beyond 80 m dropped |
| ground plane | train ∪ val road height −0.324 m in the ego frame (origin at the rear axle); camera 1.84 m above the road; the registered z = 0 plane was replaced before scoring (amendment) |
| recall comparison with KITTI | different matching and distance definitions; a caveat, not a pass/fail |

## 5. Limitations

* **Counts are small.**
  * IDM's safety harm rate is 1 of 6 states.
  * Several bootstrap intervals span most of [0, 1].
  * The s1 "consistent" outcome rests on 5 IDM states with ρ 112.
* **The evaluation is open-loop per state,** with per-frame detection and no tracking, so false positives flicker
  across the 4-frame history buffer.
* **Occluded and distant logged tracks are removed in both modes.**
  * About 8 pedestrians per state project into the image; most are behind barriers or other road users.
  * Static classes (cones, barriers, signs) pass through unchanged.
* **Lidar tracks serve as ground truth,** and image–lidar offsets up to 41 ms are not compensated.
* **Camera 2–8 URLs follow the user's naming rule.** Each was verified by HEAD and by its directory's log set.

## 6. What this changes for the paper

* **Report the nuPlan track with real perception.** The transported-profile numbers overstate how much escalation
  matters for both planners, and overstate IDM's sign variation. Keep them only as a labelled sensitivity.
* **The external replication of sign-varying value holds for PDM-Closed, not for both planners.**
  * PDM-Closed: benchmark-level harm rate and ρ, and a non-trivial oracle@20 reduction (14.7% on safety, 39.4% on
    the test logs).
  * IDM: nearly insensitive to the fidelity choice.
  * Under the registered rule, this is "intermediate", not "consistent". The claim should be stated per planner.
* **Planner identity matters even more than Track B suggested.** The same detections produce meaningful, sign-varying
  value for one published planner and almost none for the other.
* **False positives are a first-class part of the real-perception mechanism on this track.** A benchmark variant
  that models only misses, as the transported profile did, would miss most of PDM-Closed's value.
