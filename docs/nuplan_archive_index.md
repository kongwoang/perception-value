# nuPlan sensor archive index — feasibility (metadata only)

Pre-registration: `RESEARCH_LOG.md`, 2026-09-14 "Task 4 pre-registration" (commit `3bf1960`, before any
request). Script: `scripts/111_nuplan_archive_index.py`; `--mode report` does the mapping and pricing
offline. Index: `results/final/nuplan_archive_index.csv`. Inputs and working files:
`results/raw/nuplan_archive_urls/`. Request logs: `results/raw/2026091414{0007,1715,1744,1856}_nuplan_archive_index/requests.jsonl`.

Units: GB = 10^9 bytes, MB = 10^6. The download page's "GB" are GiB: Camera 0 is 52,219,710,368 bytes by
HEAD, which is 48.63 GiB, as displayed.

## Verdict

**The 9-test-log real-perception run cannot be done in full now, but a third of it can, and the rest
needs three links and one permission from you.**

The test logs' CAM_F0 images sit in four of the nine v1.1 mini camera shards: Camera 0, 2, 3 and 6.

* **Camera 0 is ready.**
  * Its official link is a public S3 object. HEAD and Range work without login.
  * I have read its full central directory: 242,385 members, 42 MB.
  * It holds exactly the seven logs of metadata "File group 0", three of them test logs.
  * The 769 scenario-window images of those three logs (167 MB), or all 11,060 of their CAM_F0 images
    (2.28 GB), could be fetched now by Range without downloading the 52 GB archive. I did not fetch them,
    because Task 4 allows no data download.
* **The other six test logs need Cameras 2, 3 and 6.** For those shards you need to:
  1. Send me the official links for Camera 2, 3 and 6. I did not construct them from Camera 0's naming
     pattern.
  2. Allow me to read their central directories. That is about 40 MB each, ~120 MB in total, beyond the
     50 MB metadata cap; 42.0 MB of the cap is used and 10.4 MB is left.
  3. Allow the data download itself.
* **Recommended: option (c) via Range, 0.83 GB in total.**
  * All 3,320 test-window images: 0.71 GB of images plus 0.12 GB of directories.
  * The alternative is option (a): four whole shards, 202 GB of transfer. They cannot be held together in
    the 114.6 GB free, but each fits one at a time if deleted after extraction.

The measured mean CAM_F0 JPEG is **211.0 kB stored, 212.1 kB uncompressed**. The Task 3 estimate of
212 kB therefore stands; no Task 3 figure changes by more than 0.5%.

## What was fetched

Total **42,019,619 bytes** in 18 requests, against a 52,428,800-byte cap. No archive member, image or
lidar byte was fetched.

| run | requests | bytes | what |
|---|---|---|---|
| 140007 | GET nuscenes.org/nuplan | 6,574 | JS page, no links; download table behind login and Terms of Use (stopped, reported) |
| 141715 | 2 HEAD | 0 | metadata txt: 2,622 B, `Accept-Ranges: bytes`. Camera 0: 52,219,710,368 B, `application/zip`, `Accept-Ranges: bytes`, 200 with no login |
| 141744 | GET metadata txt | 2,622 | 64 mini logs in File groups 0–8 |
| 141856 | HEAD + 13 Range GET | 42,010,423 | Camera 0: last 65,557 B (EOCD), ZIP64 locator and record, then the central directory (41,944,810 B) in chunks; every Range returned 206 |

**URL sources.** You copied both links from the logged-in official download page:
* "Mini Sensors Metadata" → `https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan_mini_sensor.txt`
* "Camera 0" → `https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/nuplan-v1.1_mini_camera_0.zip`

Camera 1–8 names and sizes come from the page text you transcribed; their links are unknown.

## Steps as registered

| step | result |
|---|---|
| 1. Archive names and URLs from official sources | devkit: none, only "create an account and agree to the Terms of Use". Website: behind login. You supplied the page text (v1.1 Mini Sensors: Metadata, Camera 0–8, Lidar 0–8) and the Metadata and Camera 0 links |
| 2. HEAD sizes; Range; central directory | Metadata 2,622 B. Camera 0 52.22 GB, Range supported without login, central directory read in full: ZIP64, 242,385 entries: 242,320 JPEGs (all deflated; 8 cameras × 30,290), 64 directories and one LICENSE file |
| 3. Logs → archives | Metadata File group *i* lists the logs of shard *i*. Verified for shard 0: the logs in the directory equal File group 0 exactly. For shards 1–8 this rests on the metadata file and the matching "Camera *i*" names, and is checked when a directory is read. The 9 test logs fall in groups 0, 2, 3, 6; the 34 benchmark logs in all nine |
| 4. Download options vs free disk | below |
| Measured mean JPEG | CAM_F0 in Camera 0 (30,290 images): 211,049 B stored (median 212,775), 212,086 B uncompressed. All eight cameras: 215,176 B stored |

## Needed images per archive

Camera 0 bytes are exact: every needed file name was joined to its member, and all were found. Other shards
are estimated as images × 211,049 B. Sizes for Camera 1–8 are the displayed GiB.

| archive | size | benchmark logs | test logs | all 34 logs, whole log | all 34, scenario windows | 9 test logs, whole log | 9 test, windows |
|---|---|---|---|---|---|---|---|
| Camera 0 | 52.22 GB | 7 | 3 | 30,290 · 6.39 GB | 2,308 · 483.6 MB | 11,060 · 2.28 GB | 769 · 166.7 MB |
| Camera 1 | 54.20 GB | 3 | 0 | 13,070 · 2.76 GB | 1,539 · 324.8 MB | — | — |
| Camera 2 | 49.96 GB | 4 | 2 | 16,870 · 3.56 GB | 1,758 · 371.0 MB | 9,180 · 1.94 GB | 660 · 139.3 MB |
| Camera 3 | 49.94 GB | 1 | 1 | 4,600 · 0.97 GB | 220 · 46.4 MB | 4,600 · 0.97 GB | 220 · 46.4 MB |
| Camera 4 | 49.16 GB | 3 | 0 | 11,870 · 2.51 GB | 880 · 185.7 MB | — | — |
| Camera 5 | 50.79 GB | 4 | 0 | 20,009 · 4.22 GB | 1,100 · 232.2 MB | — | — |
| Camera 6 | 49.90 GB | 4 | 3 | 15,910 · 3.36 GB | 1,891 · 399.1 MB | 12,160 · 2.57 GB | 1,671 · 352.7 MB |
| Camera 7 | 49.34 GB | 5 | 0 | 22,170 · 4.68 GB | 2,346 · 495.1 MB | — | — |
| Camera 8 | 45.16 GB | 3 | 0 | 10,150 · 2.14 GB | 879 · 185.5 MB | — | — |
| **total** | **450.66 GB** | **34** | **9** | **144,939 · 30.59 GB** | **12,921 · 2.72 GB** | **37,000 · 7.76 GB** | **3,320 · 705.1 MB** |

## Three download options

Free disk now: **114.6 GB** (106.7 GiB).

| option | 9 test logs | all 34 benchmark logs | against free disk |
|---|---|---|---|
| **(a) whole archives** | Cameras 0, 2, 3, 6: **202.0 GB** of transfer (188.1 GiB); largest 52.2 GB | all nine: **450.7 GB** (419.7 GiB); largest 54.2 GB | not together. One shard at a time fits: download, extract the needed CAM_F0, delete. Peak ≈ one shard plus the extracted images |
| **(b) needed CAM_F0 members via Range** (whole logs) | 37,000 images, **7.76 GB**, plus 120 MB of unread directories (Cameras 2, 3, 6) | 144,939 images, **30.6 GB**, plus 320 MB of unread directories | fits |
| **(c) scenario-window images only** (via Range) | 3,320 images, **0.71 GB**, plus the same 120 MB of directories | 12,921 images, **2.72 GB**, plus 320 MB | fits |

Local ZIP headers add about 142 B per member, 5 MB for (b) on the test logs.

**Is (b)/(c) technically possible?** Yes, for Camera 0, verified.
* The server advertises `Accept-Ranges: bytes`, and all 13 Range requests returned 206 with exactly the
  requested bytes.
* Each member has a local-header offset and a compressed size in the central directory. A member is
  therefore one Range request: its local header plus its deflate stream, inflated locally. No member was
  fetched to test this, since that would be data.
* For Cameras 2, 3 and 6 the same bucket and path are likely but unverified until their links are HEADed.

**A cheaper directory read is possible but untested.** Camera 0's directory is ordered by log, then
camera: every log-and-camera folder is one contiguous run, apart from the root entry and a trailing LICENSE.
Its local offsets increase monotonically. Sampling small Range chunks
could locate just the needed CAM_F0 runs instead of reading ~40 MB per shard.

## What needs your action

| # | action | unblocks |
|---|---|---|
| 1 | Copy the official links for **Camera 2, 3, 6** from the logged-in download page. Add Camera 1, 4, 5, 7, 8 for all 34 logs | HEAD and Range checks on those shards |
| 2 | Allow central-directory reads beyond the 50 MB cap: ~120 MB for the test logs, ~320 MB for all 34 | exact member offsets, and the group-to-shard check |
| 3 | Approve the data download. Recommended: (c) for the test logs, 0.71 GB, Range fetch of CAM_F0 window images, one shard at a time, with a byte log. Whether partial Range fetches are acceptable under the Terms of Use you accepted is your call | the 9-test-log real-perception run |

Without step 2, (a) is the fallback: 202 GB of transfer, one shard on disk at a time.
