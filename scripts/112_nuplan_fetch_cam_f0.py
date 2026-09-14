#!/usr/bin/env python
"""Task 5 Stage A: fetch only the CAM_F0 scenario-window images of the 34 benchmark logs, by HTTP Range.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44, "Task 5 pre-registration").  Nothing but metadata
and the planned JPEG members is ever requested:

  --stage dirs   per shard 0-8: HEAD, end-of-central-directory (ZIP64), full central directory by Range;
                 verifies that the shard's logs equal metadata File group i.  Cap 400 MB for this stage.
  --stage plan   joins the window images to the directories; exact planned bytes; cap 3.5 GB.
  --stage fetch  one shard at a time: one Range per member (local header + deflate stream), inflate,
                 CRC32 against the directory, JPEG decode to 1080x1920x3, atomic write.  Resumable.

A Range request answered with anything but 206 is never read: the connection is closed, so a server that
ignores Range cannot turn a member request into an archive download.  Every request is logged with its
byte count under results/raw/nuplan_task5/.
"""
from __future__ import annotations

import argparse, gzip, http.client, json, re, struct, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URLS_DIR = ROOT / "results" / "raw" / "nuplan_archive_urls"
WORK = ROOT / "results" / "raw" / "nuplan_task5"
OUT_ROOT = Path("/home/kongwoang/datasets/nuplan/sensor_blobs_cam_f0")
CD_CAP = 400_000_000
IMG_CAP = 3_500_000_000
UA = "risk-aware-perception-task5/1.0 (CAM_F0 window members only)"
CAMERA1_URL = ("https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/mini_set/"
               "nuplan-v1.1_mini_camera_1.zip")
SRC0 = "official download page (logged in), 'Camera 0' link, copied by the user 2026-09-14 (Task 4)"
SRC1 = "official download page (logged in), 'Camera 1' link, pasted by the user with Task 5, 2026-09-14"
SRC_RULE = ("built from the Camera 1 link by the user's stated rule, Task 5 message 2026-09-14: "
            "'thay số là ra link nên tôi ko copy nữa' (replace the number and you get the link)")


def shard_urls() -> dict[int, tuple[str, str]]:
    cam0 = [l.split("\t") for l in (URLS_DIR / "user_provided.tsv").read_text().splitlines()
            if "_camera_0.zip\t" in l][0]
    urls = {0: (cam0[0], SRC0), 1: (CAMERA1_URL, SRC1)}
    for i in range(2, 9):
        urls[i] = (CAMERA1_URL.replace("_camera_1.zip", f"_camera_{i}.zip"), SRC_RULE)
    return urls


def file_groups() -> dict[str, int]:
    group, g = {}, None
    for line in (URLS_DIR / "nuplan_mini_sensor.txt").read_text().splitlines():
        m = re.match(r"File group:\s*(\d+)", line.strip())
        if m:
            g = int(m.group(1))
        elif line.strip():
            group[line.strip()] = g
    return group


class Stop(Exception):
    pass


class Ledger:
    """Append-only request log with per-kind cumulative byte counters across every run of this task."""

    def __init__(self, stage: str):
        WORK.mkdir(parents=True, exist_ok=True)
        self.totals = {"meta": 0, "member": 0}
        for f in WORK.glob("requests_*.jsonl"):
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    self.totals[r["kind"]] = self.totals.get(r["kind"], 0) + r["body_bytes"]
        self.path = WORK / f"requests_{stage}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.lock = threading.Lock()
        print(f"  ledger: meta {self.totals['meta']:,} B, member {self.totals['member']:,} B so far")

    def reserve(self, kind: str, n: int):
        cap = CD_CAP if kind == "meta" else IMG_CAP
        with self.lock:
            if self.totals[kind] + n > cap:
                raise Stop(f"{kind} request of {n} B would pass the {cap:,} B cap ({self.totals[kind]:,} used)")
            self.totals[kind] += n

    def release(self, kind: str, n: int):
        with self.lock:
            self.totals[kind] -= n

    def log(self, rec: dict):
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


_local = threading.local()


def _conn(host: str):
    c = getattr(_local, "conn", None)
    if c is None or getattr(_local, "host", None) != host:
        c = http.client.HTTPSConnection(host, timeout=60)
        _local.conn, _local.host = c, host
    return c


def _drop_conn():
    c = getattr(_local, "conn", None)
    if c is not None:
        try:
            c.close()
        except Exception:                                                    # noqa: BLE001
            pass
    _local.conn = None


def request(led: Ledger, kind: str, method: str, url: str, rng: tuple[int, int] | None = None,
            tries: int = 3) -> tuple[int | None, dict, bytes]:
    """One logged request; a Range GET is read only when answered 206 with the exact range."""
    want = (rng[1] - rng[0] + 1) if rng else 0
    u = urlsplit(url)
    last = None
    for attempt in range(1, tries + 1):
        led.reserve(kind, want)
        rec = {"t": time.strftime("%H:%M:%S"), "kind": kind, "method": method, "url": url,
               "range": list(rng) if rng else None, "attempt": attempt}
        body, status, headers = b"", None, {}
        try:
            c = _conn(u.netloc)
            h = {"User-Agent": UA}
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            c.request(method, u.path, headers=h)
            r = c.getresponse()
            status, headers = r.status, {k.lower(): v for k, v in r.getheaders()}
            if method == "HEAD":
                r.read()
            elif rng and status == 206 and headers.get("content-range", "").startswith(
                    f"bytes {rng[0]}-{rng[1]}/"):
                body = r.read(want)
            else:
                _drop_conn()                  # never read an unexpected body
        except Exception as e:                                               # noqa: BLE001
            _drop_conn()
            rec["error"] = f"{type(e).__name__}: {e}"
        led.release(kind, want - len(body))
        rec.update(status=status, body_bytes=len(body),
                   content_length=headers.get("content-length"), content_range=headers.get("content-range"))
        led.log(rec)
        last = (status, headers, body)
        if method == "HEAD" and status is not None:
            return last
        if rng and status == 206 and len(body) == want:
            return last
        if status is not None and status < 500 and status != 206:
            return last                        # a definite refusal: do not retry
        time.sleep(2 * attempt)
    return last


# ---------------------------------------------------------------------------------------------- dirs

def eocd(led, url, size):
    tail = min(size, 65_557)
    st, _, body = request(led, "meta", "GET", url, (size - tail, size - 1))
    if st != 206:
        raise Stop(f"EOCD Range refused: status {st} for {url}")
    i = body.rfind(b"PK\x05\x06")
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", body[i:i + 22])
    info = {"entries": n_total, "cd_size": cd_size, "cd_offset": cd_off, "zip64": False}
    if 0xFFFFFFFF in (cd_off, cd_size) or n_total == 0xFFFF:
        j = body.rfind(b"PK\x06\x07", 0, i)
        _, _, rec64, _ = struct.unpack("<IIQI", body[j:j + 20])
        st, _, b2 = request(led, "meta", "GET", url, (rec64, rec64 + 55))
        if st != 206 or b2[:4] != b"PK\x06\x06":
            raise Stop(f"ZIP64 record unreadable for {url}")
        v = struct.unpack("<IQHHIIQQQQ", b2[:56])
        info.update(entries=v[7], cd_size=v[8], cd_offset=v[9], zip64=True)
    return info


def parse_cd(buf: bytes, rows: list) -> int:
    """Parse complete central-directory records from buf; returns bytes consumed."""
    p = 0
    while p + 46 <= len(buf):
        if buf[p:p + 4] != b"PK\x01\x02":
            raise Stop(f"bad central-directory signature at +{p}")
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", buf[p:p + 46])
        end = p + 46 + nlen + xlen + clen
        if end > len(buf):
            break
        name = buf[p + 46:p + 46 + nlen].decode("utf-8")
        extra = buf[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                r = q + 4
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", extra[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", extra[r:r + 8])[0]; r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", extra[r:r + 8])[0]; r += 8
            q += 4 + hlen
        rows.append((name, method, flags, crc, csize, usize, off, nlen, xlen))
        p = end
    return p


def stage_dirs(shards):
    led = Ledger("dirs")
    urls, groups = shard_urls(), file_groups()
    (WORK / "cd").mkdir(parents=True, exist_ok=True)
    (WORK / "shard_urls.tsv").write_text("".join(f"{i}\t{u}\t{s}\n" for i, (u, s) in sorted(urls.items())))
    summ_path = WORK / "dirs.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    # HEAD and EOCD of every shard first, so the directory total is known before any directory is read
    for i in shards:
        if str(i) in summ and summ[str(i)].get("cd_read"):
            continue
        url, src = urls[i]
        st, h, _ = request(led, "meta", "HEAD", url)
        rec = {"shard": i, "url": url, "source": src, "head_status": st, "size": int(h.get("content-length", 0)),
               "content_type": h.get("content-type"), "accept_ranges": h.get("accept-ranges"),
               "last_modified": h.get("last-modified"), "login_required": st in (401, 403)}
        print(f"  HEAD camera_{i}: {st} {rec['size']:,} B {rec['content_type']} ranges={rec['accept_ranges']}")
        if st != 200 or rec["content_type"] != "application/zip" or rec["accept_ranges"] != "bytes":
            summ[str(i)] = rec
            summ_path.write_text(json.dumps(summ, indent=1))
            raise Stop(f"camera_{i}: HEAD {st}, type {rec['content_type']}, ranges {rec['accept_ranges']}")
        rec.update(eocd(led, url, rec["size"]))
        summ[str(i)] = rec
        summ_path.write_text(json.dumps(summ, indent=1))
    pending = [i for i in shards if not summ[str(i)].get("cd_read")]
    need = sum(summ[str(i)]["cd_size"] for i in pending
               if not (WORK / "cd" / f"members_camera_{i}.csv.gz").exists())
    print(f"  central directories still to read: {len(pending)} shards, {need:,} B; "
          f"meta used {led.totals['meta']:,} of {CD_CAP:,}")
    if led.totals["meta"] + need > CD_CAP:
        raise Stop(f"directories need {need:,} B beyond {led.totals['meta']:,} B used: over the 400 MB cap")
    for i in pending:
        rec = summ[str(i)]
        done_csv = WORK / "cd" / f"members_camera_{i}.csv.gz"
        rows, buf, pos, end = [], b"", rec["cd_offset"], rec["cd_offset"] + rec["cd_size"]
        if done_csv.exists():
            # a directory already read in full by an earlier run is never fetched again
            m = pd.read_csv(done_csv)
            if len(m) != rec["entries"]:
                raise Stop(f"camera_{i}: saved directory has {len(m)} of {rec['entries']} entries")
            pos = end
        while pos < end:
            b = min(pos + 8_000_000, end) - 1
            st, _, body = request(led, "meta", "GET", rec["url"], (pos, b))
            if st != 206:
                raise Stop(f"camera_{i}: directory Range refused ({st})")
            buf += body
            pos = b + 1
            buf = buf[parse_cd(buf, rows):]
        if not done_csv.exists() and (buf or len(rows) != rec["entries"]):
            raise Stop(f"camera_{i}: parsed {len(rows)} of {rec['entries']} entries, {len(buf)} B left over")
        if not done_csv.exists():
            m = pd.DataFrame(rows, columns=["name", "method", "flags", "crc32", "compressed", "uncompressed",
                                            "local_offset", "name_len", "extra_len"])
            m.to_csv(done_csv, index=False)
        parts = m["name"].str.split("/")
        logs = set(parts[parts.str.len() >= 3].str[1])
        want = {k for k, v in groups.items() if v == i}
        f0 = m[m["name"].str.contains("/CAM_F0/") & m["name"].str.endswith(".jpg")]
        rec.update(cd_read=True, members=len(m), jpg_members=int(m["name"].str.endswith(".jpg").sum()),
                   cam_f0_members=len(f0), logs=len(logs), logs_equal_file_group=logs == want,
                   methods={str(k): int(v) for k, v in m.method.value_counts().items()}, data_descriptor_flag=int((m["flags"] & 8).gt(0).sum()),
                   cam_f0_mean_compressed=float(f0.compressed.mean()))
        summ[str(i)] = rec
        summ_path.write_text(json.dumps(summ, indent=1, default=int))
        print(f"  camera_{i}: {len(m):,} entries, {len(f0):,} CAM_F0, {len(logs)} logs, "
              f"equal to File group {i}: {logs == want}; meta used {led.totals['meta']:,} B")
        if logs != want:
            raise Stop(f"camera_{i}: logs differ from File group {i}: "
                       f"missing {sorted(want - logs)}, extra {sorted(logs - want)}")


# ---------------------------------------------------------------------------------------------- plan

def stage_plan():
    need = pd.read_csv(URLS_DIR / "needed_cam_f0.csv.gz")
    need = need[need.in_window == 1].copy()
    parts = []
    for i in range(9):
        m = pd.read_csv(WORK / "cd" / f"members_camera_{i}.csv.gz")
        m["filename_jpg"] = m["name"].str.split("/", n=1).str[1]
        m["shard"] = i
        parts.append(m[m["name"].str.contains("/CAM_F0/")])
    mem = pd.concat(parts, ignore_index=True)
    j = need.merge(mem, on="filename_jpg", how="left", validate="one_to_one")
    missing = int(j["name"].isna().sum())
    if missing:
        raise Stop(f"{missing} window images are in no shard's directory")
    wrong = int((j.shard != j.group).sum())
    j["planned_bytes"] = 30 + j.name_len + j.extra_len + j.compressed
    total = int(j.planned_bytes.sum())
    print(f"  {len(j):,} window images, {j.log.nunique()} logs, shard != group for {wrong}; "
          f"planned {total:,} B (cap {IMG_CAP:,}); methods {j.method.value_counts().to_dict()}")
    if wrong:
        raise Stop(f"{wrong} images sit in a shard other than their File group")
    if total > IMG_CAP:
        raise Stop(f"planned {total:,} B exceeds the 3.5 GB cap")
    cols = ["log", "split", "shard", "filename_jpg", "name", "method", "flags", "crc32", "compressed",
            "uncompressed", "local_offset", "name_len", "extra_len", "planned_bytes"]
    j[cols].sort_values(["shard", "local_offset"]).to_csv(WORK / "plan_members.csv.gz", index=False)
    (WORK / "plan.json").write_text(json.dumps(
        {"images": len(j), "logs": int(j.log.nunique()), "planned_bytes": total, "cap": IMG_CAP,
         "per_shard": j.groupby("shard").agg(images=("name", "size"), bytes=("planned_bytes", "sum"))
         .reset_index().to_dict("records")}, indent=1, default=int))


# ---------------------------------------------------------------------------------------------- fetch

def fetch_one(led, url, r):
    off, nlen, xlen, csize = int(r.local_offset), int(r.name_len), int(r.extra_len), int(r.compressed)
    first = 30 + nlen + xlen + csize
    st, _, body = request(led, "member", "GET", url, (off, off + first - 1))
    if st != 206:
        raise Stop(f"member Range refused ({st}) at {r.name}")
    sig, _, flags, method, _, _, crc_l, _, _, nlen_l, xlen_l = struct.unpack("<IHHHHHIIIHH", body[:30])
    if sig != 0x04034B50 or body[30:30 + nlen_l].decode("utf-8") != r.name:
        raise RuntimeError(f"local header mismatch at {r.name}")
    start = 30 + nlen_l + xlen_l
    extra_req = 0
    if start + csize > len(body):
        st, _, more = request(led, "member", "GET", url, (off + len(body), off + start + csize - 1))
        if st != 206:
            raise Stop(f"member remainder Range refused ({st}) at {r.name}")
        body += more
        extra_req = 1
    overread = len(body) - (start + csize)
    comp = body[start:start + csize]
    if int(r.method) == 8:
        d = zlib.decompressobj(-15)
        data = d.decompress(comp) + d.flush()
    elif int(r.method) == 0:
        data = comp
    else:
        raise RuntimeError(f"method {r.method} at {r.name}")
    crc_ok = (zlib.crc32(data) & 0xFFFFFFFF) == int(r.crc32) and len(data) == int(r.uncompressed)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if crc_ok else None
    dec_ok = img is not None and img.shape == (1080, 1920, 3)
    if crc_ok and dec_ok:
        dst = OUT_ROOT / r.filename_jpg
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".jpg.part")
        tmp.write_bytes(data)
        tmp.replace(dst)
    return {"filename_jpg": r.filename_jpg, "shard": int(r.shard), "fetched_bytes": len(body),
            "extra_requests": extra_req, "overread_bytes": max(overread, 0), "crc_ok": crc_ok,
            "decode_ok": dec_ok, "local_crc_matches": crc_l in (0, int(r.crc32))}


def stage_fetch(shards, workers):
    led = Ledger("fetch")
    urls = shard_urls()
    plan = pd.read_csv(WORK / "plan_members.csv.gz")
    man_path = WORK / "manifest.csv"
    done = set()
    if man_path.exists():
        man = pd.read_csv(man_path)
        good = man[man.crc_ok & man.decode_ok]
        done = {f for f in good.filename_jpg if (OUT_ROOT / f).exists()}
    left = int(plan[~plan.filename_jpg.isin(done)].planned_bytes.sum())
    print(f"  {len(done):,} of {len(plan):,} already verified; {left:,} B left; member bytes used "
          f"{led.totals['member']:,} of {IMG_CAP:,}")
    if led.totals["member"] + left > IMG_CAP:
        raise Stop("remaining planned bytes would pass the 3.5 GB cap")
    lock = threading.Lock()
    header = not man_path.exists()
    for i in shards:
        todo = plan[(plan.shard == i) & ~plan.filename_jpg.isin(done)]
        if todo.empty:
            continue
        t0, n_ok, n_bad, nbytes = time.time(), 0, 0, 0
        print(f"  camera_{i}: {len(todo):,} members, {int(todo.planned_bytes.sum()):,} B", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(fetch_one, led, urls[i][0], r) for r in todo.itertuples()]
            for k, f in enumerate(as_completed(futs), 1):
                rec = f.result()                      # Stop and hard errors propagate here
                with lock, open(man_path, "a") as fh:
                    if header:
                        fh.write(",".join(rec) + "\n"); header = False
                    fh.write(",".join(str(v) for v in rec.values()) + "\n")
                n_ok += rec["crc_ok"] and rec["decode_ok"]
                n_bad += not (rec["crc_ok"] and rec["decode_ok"])
                nbytes += rec["fetched_bytes"]
                if k % 500 == 0 or k == len(futs):
                    dt = time.time() - t0
                    print(f"    {k:,}/{len(futs):,} ok {n_ok:,} bad {n_bad} {nbytes / 1e6:,.0f} MB "
                          f"{nbytes / 1e6 / max(dt, 1e-9):.1f} MB/s", flush=True)
        if n_bad:
            raise Stop(f"camera_{i}: {n_bad} members failed CRC or decode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dirs", "plan", "fetch"])
    ap.add_argument("--shards", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    shards = [int(s) for s in args.shards.split(",")]
    try:
        {"dirs": lambda: stage_dirs(shards), "plan": stage_plan,
         "fetch": lambda: stage_fetch(shards, args.workers)}[args.stage]()
    except Stop as e:
        (WORK / "stop.json").write_text(json.dumps({"stage": args.stage, "t": time.strftime("%F %T"),
                                                    "reason": str(e)}, indent=1))
        print(f"\n  STOP: {e}")
        sys.exit(2)
    print("  done")


if __name__ == "__main__":
    main()
