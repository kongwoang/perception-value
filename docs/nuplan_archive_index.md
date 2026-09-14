# nuPlan sensor archive index — feasibility (metadata only)

Pre-registration: `RESEARCH_LOG.md`, 2026-09-14 "Task 4 pre-registration" (commit `3bf1960`, before any
request). Script: `scripts/111_nuplan_archive_index.py`. Request log:
`results/raw/20260914_140007_nuplan_archive_index/requests.jsonl`.

## Verdict

**The 9-test-log real-perception run cannot be done now, and the archive index needs your action.**

The only official sources for archive names and URLs are the local nuplan-devkit docs and the nuPlan
website. The devkit (`docs/dataset_setup.md`) says that to download nuPlan you must *"create an account
and agree to the Terms of Use. After logging in you will see multiple archives."* It lists no archive
names or URLs. The website `https://www.nuscenes.org/nuplan` is a JavaScript application whose
server-rendered page carries no download links. The download table is shown only to logged-in users, so
the index sits behind a login and a terms page. Per the rules, the inquiry stops there. No bundle or API
was mined for links, and no archive was touched. **Total bytes fetched: 6,574** (one HTML page), against
a 50 MB cap.

What is known without the index:

* **Needed images, exact** (from the local mini DBs, Task 3).
  * 9 test logs: 3,320 CAM_F0 images inside the scenario windows; 37,000 in the whole logs.
  * All 34 benchmark logs: 12,921 and 144,939.
* **Needed download under the scenario-windows option, still an estimate.**
  * 9 test logs: ~0.70 GB.
  * All 34 logs: ~2.7 GB.
  * Based on Task 3's 212 KB per image. It cannot be replaced by a measured mean without a central
    directory.
* **Whole-archive option.** An earlier session recorded, in `third_party/PROVENANCE.md` and without
  URLs, that the v1.1 mini camera data is nine shards of 45–54 GB (~450 GB), split by blob rather than
  by log. If that holds, whole shards do not fit in the ~107 GB currently free. Which shards hold our logs
  is unknown until their central directories are read.

**What I need from you:**
1. Log in at `https://www.nuscenes.org/nuplan#download`, creating an account and accepting the Terms of
   Use if you have not.
2. Send me the names and links of the v1.1 **mini sensor/camera** archives shown there. Say whether the
   links are signed or expire. A saved copy of the page, or the copied link list, is enough.

With those official URLs I can finish steps 2–4 within the 50 MB metadata cap, provided the server allows
HEAD and Range without further login:
* archive sizes;
* whether Range is supported;
* the central-directory listing and the exact shards holding each of the 34 and 9 logs;
* the measured JPEG size;
* the three download options against free disk.

## Steps as registered

| step | result |
|---|---|
| 1. Local devkit: archive names and URLs | none listed. Only the download page, and the statement that login and Terms of Use are required |
| 1b. Official website | `https://www.nuscenes.org/nuplan` → 200, 6,574 bytes, 10 links, 0 archive links; JavaScript-rendered |
| 2. HEAD / Range / central directory | not reached: no archive URL from an official source without login |
| 3. Map logs to archives | not possible without an index. Earlier record: shards split by blob, not by log |
| 4a. Whole archives | unknown which. Upper bound ~450 GB for all nine mini camera shards (earlier record) vs ~107 GB free → does not fit |
| 4b. Needed CAM_F0 members via Range | cannot be assessed until an archive URL is known |
| 4c. Scenario-window images only | 3,320 images (test) / 12,921 (all 34); ~0.70 / ~2.7 GB by estimate |
| Measured mean JPEG size | not available (no central directory reachable) |
