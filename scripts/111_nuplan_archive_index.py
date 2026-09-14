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
"""
from __future__ import annotations

import argparse, csv, glob, gzip, json, re, struct, sys, time, urllib.error, urllib.request
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["page", "head", "cd"])
    ap.add_argument("--urls", default=None)
    ap.add_argument("pages", nargs="*")
    args = ap.parse_args()
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
