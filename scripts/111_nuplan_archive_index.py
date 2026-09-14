#!/usr/bin/env python
"""Task 4: how nuPlan distributes the CAM_F0 images the benchmark needs -- metadata only.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 4 pre-registration").  Every request goes through
`Fetcher`, which logs method, URL, status, headers and body bytes to requests.jsonl and refuses any request
that would take the cumulative total past 50 MB (counted across all runs of this script).  Allowed:
official pages, HEAD, and Range reads of a ZIP's end-of-central-directory record and central directory.
No credentials, tokens, account creation or terms acceptance: a 401/403, or a redirect to a login or terms
page, is recorded as `login_required` and the inquiry stops there.

  --mode page  URL [URL ...]         fetch official pages; list every link that looks like an archive
  --mode head  --urls FILE           HEAD each archive URL (FILE rows: url<TAB>source)
  --mode cd    --urls FILE           HEAD, then EOCD and central directory via Range, members to csv.gz
  --mode report                      no network: map needed CAM_F0 images to archives, price download options
"""
from __future__ import annotations

import argparse, csv, glob, gzip, json, re, struct, sys, time, urllib.error, urllib.request
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402

CAP = 50 * 1024 * 1024
UA = {"User-Agent": "risk-aware-perception-feasibility/1.0 (metadata only)"}
LOGIN_HINTS = ("login", "signin", "sign-in", "auth", "terms", "register", "account")


class Budget(Exception):
    pass


class Fetcher:
    def __init__(self, run: Path):
        self.log = run / "requests.jsonl"
        prior = 0
        for f in glob.glob(str(ROOT / "results" / "raw" / "*_nuplan_archive_index" / "requests.jsonl")):
            prior += sum(json.loads(l)["body_bytes"] for l in open(f) if l.strip())
        self.total = prior
        print(f"  bytes already fetched by earlier runs: {prior}")

    def request(self, method: str, url: str, rng: tuple[int, int] | None = None, max_body: int = 2_000_000):
        want = (rng[1] - rng[0] + 1) if rng else (0 if method == "HEAD" else max_body)
        if self.total + want > CAP:
            raise Budget(f"request would exceed the 50 MB cap ({self.total} + {want})")
        headers = dict(UA)
        if rng:
            headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
        req = urllib.request.Request(url, method=method, headers=headers)
        rec = {"t": time.strftime("%H:%M:%S"), "method": method, "url": url, "range": rng}
        body = b""
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rec.update(status=r.status, final_url=r.geturl(), headers=dict(r.headers.items()))
                if method != "HEAD":
                    body = r.read(want if rng else max_body)
        except urllib.error.HTTPError as e:
            rec.update(status=e.code, final_url=e.geturl(), headers=dict(e.headers.items()) if e.headers else {})
        except Exception as e:                                                   # noqa: BLE001
            rec.update(status=None, error=f"{type(e).__name__}: {e}")
        rec["body_bytes"] = len(body)
        self.total += len(body)
        rec["cumulative_bytes"] = self.total
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec, body


def login_required(rec) -> bool:
    fu = (rec.get("final_url") or "").lower()
    return rec.get("status") in (401, 403) or any(h in fu for h in LOGIN_HINTS) and fu != rec["url"].lower()


def mode_page(fx: Fetcher, run: Path, urls):
    out = []
    for u in urls:
        rec, body = fx.request("GET", u)
        text = body.decode("utf-8", "replace")
        (run / (re.sub(r"[^A-Za-z0-9]+", "_", u)[:80] + ".html")).write_text(text)
        links = sorted(set(re.findall(r"""(?:href|src)=["']([^"']+)["']""", text)))
        arch = [l for l in links if re.search(r"\.(zip|tar|tgz|gz)(\?|$)|sensor|blob|amazonaws|download", l, re.I)]
        print(f"  {u}: status {rec.get('status')} final {rec.get('final_url')} {len(body)} bytes, "
              f"{len(links)} links, {len(arch)} archive-like")
        for l in arch:
            print("    ", l)
        out.append({"page": u, "status": rec.get("status"), "final_url": rec.get("final_url"),
                    "bytes": len(body), "archive_links": arch,
                    "login_required": login_required(rec)})
    (run / "pages.json").write_text(json.dumps(out, indent=1))


def _eocd(fx, url, size):
    tail = min(size, 65_557)
    rec, body = fx.request("GET", url, rng=(size - tail, size - 1))
    if rec.get("status") != 206:
        return rec, None
    i = body.rfind(b"PK\x05\x06")
    if i < 0:
        return rec, None
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", body[i:i + 22])
    info = {"entries": n_total, "cd_size": cd_size, "cd_offset": cd_off, "zip64": False}
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:
        j = body.rfind(b"PK\x06\x07", 0, i)
        if j >= 0:
            (_, _, rec64_off, _) = struct.unpack("<IIQI", body[j:j + 20])
            r2, b2 = fx.request("GET", url, rng=(rec64_off, rec64_off + 55))
            if r2.get("status") == 206 and b2[:4] == b"PK\x06\x06":
                vals = struct.unpack("<IQHHIIQQQQ", b2[:56])
                info.update(entries=vals[7], cd_size=vals[8], cd_offset=vals[9], zip64=True)
    return rec, info


def _parse_cd(buf: bytes):
    rows, p = [], 0
    while p + 46 <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, usize, nlen, xlen, clen, _, _, _, off) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", buf[p:p + 46])
        if p + 46 + nlen + xlen + clen > len(buf):
            break
        name = buf[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = buf[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for need in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                    if need:
                        vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                    else:
                        vals.append(None)
                usize = vals[0] if vals[0] is not None else usize
                csize = vals[1] if vals[1] is not None else csize
                off = vals[2] if vals[2] is not None else off
            q += 4 + hlen
        rows.append((name, method, csize, usize, off))
        p += 46 + nlen + xlen + clen
    return rows, p


def mode_cd(fx: Fetcher, run: Path, url_file: str, listing: bool):
    results = []
    for line in open(url_file):
        if not line.strip() or line.startswith("#"):
            continue
        url, source = line.rstrip("\n").split("\t", 1)
        name = url.split("?")[0].rsplit("/", 1)[-1]
        rec, _ = fx.request("HEAD", url)
        h = {k.lower(): v for k, v in (rec.get("headers") or {}).items()}
        r = {"archive": name, "url": url, "source": source, "status": rec.get("status"),
             "size": int(h["content-length"]) if "content-length" in h else None,
             "login_required": login_required(rec), "range_supported": h.get("accept-ranges", "").lower() == "bytes",
             "members_listed": 0, "cd_complete": False}
        print(f"  HEAD {name}: {rec.get('status')} size {r['size']} ranges {r['range_supported']} login {r['login_required']}")
        if listing and r["size"] and not r["login_required"] and rec.get("status") == 200:
            try:
                erec, info = _eocd(fx, url, r["size"])
                r["range_supported"] = r["range_supported"] or erec.get("status") == 206
                if info:
                    r.update(cd_entries=info["entries"], cd_bytes=info["cd_size"], zip64=info["zip64"])
                    rows, got, pos, carry = [], 0, info["cd_offset"], b""
                    end = info["cd_offset"] + info["cd_size"]
                    with gzip.open(run / f"members__{name}.csv.gz", "wt", newline="") as gz:
                        w = csv.writer(gz)
                        w.writerow(["name", "method", "compressed", "uncompressed", "local_offset"])
                        while pos < end:
                            chunk_end = min(pos + 4_000_000, end) - 1
                            crec, chunk = fx.request("GET", url, rng=(pos, chunk_end))
                            if crec.get("status") != 206:
                                break
                            parsed, used = _parse_cd(carry + chunk)
                            w.writerows(parsed)
                            got += len(parsed)
                            carry = (carry + chunk)[used:]
                            pos = chunk_end + 1
                    r["members_listed"] = got
                    r["cd_complete"] = got == info["entries"]
            except Budget as e:
                r["stopped"] = str(e)
                print("   ", e)
        results.append(r)
    (run / "archives.json").write_text(json.dumps(results, indent=1))
    print(f"  total bytes fetched (all runs): {fx.total}")


# sizes shown on the logged-in download page, as transcribed by the user (GiB); Camera 0 is measured by HEAD
DISPLAYED_GIB = {0: 48.63, 1: 50.48, 2: 46.53, 3: 46.51, 4: 45.78, 5: 47.30, 6: 46.47, 7: 45.95, 8: 42.06}
URLS_DIR = ROOT / "results" / "raw" / "nuplan_archive_urls"


def mode_report():
    """No network: map needed CAM_F0 images to archives and price the three download options."""
    import re, shutil
    import pandas as pd
    txt = (URLS_DIR / "nuplan_mini_sensor.txt").read_text()
    group, g = {}, None
    for line in txt.splitlines():
        mm = re.match(r"File group:\s*(\d+)", line.strip())
        if mm:
            g = int(mm.group(1)); continue
        if line.strip():
            group[line.strip()] = g
    need = pd.read_csv(URLS_DIR / "needed_cam_f0.csv.gz")
    runs = sorted(glob.glob(str(ROOT / "results/raw/*_nuplan_archive_index/members__nuplan-v1.1_mini_camera_0.zip.csv.gz")))
    mem = pd.read_csv(runs[-1])
    arch0 = json.loads((Path(runs[-1]).parent / "archives.json").read_text())[0]
    mem = mem[mem.name.str.endswith(".jpg")].copy()
    mem["filename_jpg"] = mem.name.str.split("/", n=1).str[1]
    f0 = mem[mem.filename_jpg.str.contains("/CAM_F0/")]
    mean_c = float(f0.compressed.mean())
    local_overhead = float((30 + f0.name.str.len() + 20).mean())      # local header + name + ZIP64 extra, per member
    j = need.merge(mem[["filename_jpg", "compressed"]], on="filename_jpg", how="left")
    exact = j.group == 0
    assert j.loc[exact, "compressed"].notna().all(), "a needed group-0 image is missing from Camera 0"
    j["bytes"] = np.where(exact, j.compressed, mean_c)
    j["exact"] = exact
    free = shutil.disk_usage("/home/kongwoang").free
    urls = dict(l.rstrip("\n").split("\t", 1) for l in open(URLS_DIR / "user_provided.tsv") if l.strip())
    rows = []
    for gi in range(9):
        x = j[j.group == gi]
        name = f"nuplan-v1.1_mini_camera_{gi}.zip"
        url = next((u for u in urls if u.endswith(name)), "")
        logs = sorted(k for k, v in group.items() if v == gi)
        bench = sorted(set(x.log))
        rows.append({
            "archive": name if gi == 0 else f"Camera {gi} (file name not confirmed)",
            "url": url or "unknown (link not provided)",
            "source": urls.get(url, "official download page as transcribed by the user (name and size only)"),
            "size": arch0["size"] if gi == 0 else int(DISPLAYED_GIB[gi] * 2**30),
            "size_basis": "HEAD Content-Length" if gi == 0 else "displayed GiB on the download page",
            "login_required": False if gi == 0 else "unknown",
            "range_supported": True if gi == 0 else "unknown",
            "logs_covered": len(logs), "benchmark_logs": len(bench), "test_logs": int(x[x.split == "test"].log.nunique()),
            "needed_images": len(x), "needed_bytes": int(x.bytes.sum()),
            "needed_images_windows": int(x.in_window.sum()), "needed_bytes_windows": int(x[x.in_window == 1].bytes.sum()),
            "test_needed_images": int((x.split == "test").sum()), "test_needed_bytes": int(x[x.split == "test"].bytes.sum()),
            "test_needed_images_windows": int(((x.split == "test") & (x.in_window == 1)).sum()),
            "test_needed_bytes_windows": int(x[(x.split == "test") & (x.in_window == 1)].bytes.sum()),
            "bytes_basis": "exact from central directory" if gi == 0 else f"estimated at measured CAM_F0 mean {mean_c:.0f} B",
            "logs_to_archive_basis": "verified from central directory" if gi == 0 else "metadata file group; group-to-camera naming verified only for 0",
        })
    # central directory of the other shards, scaled from Camera 0 by archive size (needed for Range extraction)
    cd0 = arch0["cd_bytes"]
    for r in rows:
        r["central_directory_bytes"] = cd0 if r["size_basis"].startswith("HEAD") else int(cd0 * r["size"] / arch0["size"])
        r["central_directory_basis"] = "read" if r["size_basis"].startswith("HEAD") else "estimated from Camera 0 by size"
    txt_url = next(u for u in urls if u.endswith("nuplan_mini_sensor.txt"))
    rows.insert(0, {"archive": "nuplan_mini_sensor.txt", "url": txt_url, "source": urls[txt_url], "size": len(txt.encode()),
                    "size_basis": "HEAD Content-Length", "login_required": False, "range_supported": True,
                    "logs_covered": len(group), "benchmark_logs": 0, "test_logs": 0, "needed_images": 0, "needed_bytes": 0,
                    **{f"{c}_{u}": 0 for c in ("needed", "test_needed") for u in ("images_windows", "bytes_windows")},
                    "test_needed_images": 0, "test_needed_bytes": 0,
                    "central_directory_bytes": 0, "central_directory_basis": "not a zip",
                    "bytes_basis": "log-to-file-group index, fetched in full",
                    "logs_to_archive_basis": "lists all 64 mini logs by file group"})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "final" / "nuplan_archive_index.csv"
    df.to_csv(out, index=False)

    df_all, df = df, df[df.archive != "nuplan_mini_sensor.txt"].reset_index(drop=True)

    def opts(sub, label):
        arch = df[df[sub] > 0]
        whole = int(arch["size"].sum())
        col = "test_needed" if label == "9 test logs" else "needed"
        img, byt = int(df[f"{col}_images"].sum()), int(df[f"{col}_bytes"].sum())
        wimg, wbyt = int(df[f"{col}_images_windows"].sum()), int(df[f"{col}_bytes_windows"].sum())
        return {"set": label, "archives": ", ".join(str(i) for i in arch.index), "whole_archives_bytes": whole,
                "members_images": img, "members_bytes": byt, "members_bytes_with_headers": int(byt + img * local_overhead),
                "window_images": wimg, "window_bytes": wbyt, "window_bytes_with_headers": int(wbyt + wimg * local_overhead),
                "largest_single_archive_bytes": int(arch["size"].max()),
                "central_directories_bytes": int(arch["central_directory_bytes"].sum()),
                "central_directories_bytes_not_yet_read": int(arch.loc[arch.central_directory_basis != "read",
                                                                       "central_directory_bytes"].sum())}
    summary = {"free_bytes": free, "measured_cam_f0_mean_compressed_bytes": mean_c,
               "measured_cam_f0_mean_uncompressed_bytes": float(f0.uncompressed.mean()),
               "cam_f0_members_in_camera_0": len(f0), "local_header_overhead_bytes_mean": local_overhead,
               "options": [opts("test_logs", "9 test logs"), opts("benchmark_logs", "34 benchmark logs")]}
    (URLS_DIR / "report_summary.json").write_text(json.dumps(summary, indent=1))
    print(df_all[["archive", "size", "logs_covered", "benchmark_logs", "test_logs", "needed_images", "needed_bytes",
              "test_needed_images_windows", "test_needed_bytes_windows"]].to_string(index=False))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["page", "head", "cd", "report"])
    ap.add_argument("--urls", default=None)
    ap.add_argument("pages", nargs="*")
    args = ap.parse_args()
    if args.mode == "report":
        return mode_report()
    run = runmeta.new_run("nuplan_archive_index", vars(args))
    fx = Fetcher(run)
    try:
        if args.mode == "page":
            mode_page(fx, run, args.pages)
        else:
            mode_cd(fx, run, args.urls, listing=args.mode == "cd")
    except Budget as e:
        print("  STOP:", e)
    print("  run dir", run)


if __name__ == "__main__":
    main()
