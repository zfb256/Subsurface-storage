"""Download the TPDC daily snow depth dataset of China over FTP."""
import argparse
import ftplib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/snow/raw"


def connect(cred):
    cfg = {}
    for line in Path(cred).expanduser().read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    for attempt in range(30):
        ftp = ftplib.FTP()
        ftp.encoding = "utf-8"
        try:
            port = int(cfg.get("port", "21"))
            ftp.connect(host=cfg["host"], port=port, timeout=120)
            ftp.login(cfg["user"], cfg["password"])
            ftp.set_pasv(True)
            return ftp
        except ftplib.error_temp as e:  # e.g. "421 ... maximum users already logged in"
            print(f"server busy ({str(e)[:40]}...), retry {attempt + 1}/30 in 60 s", flush=True)
            time.sleep(60)
    raise SystemExit("FTP server still busy after 30 attempts; try again later (off-peak hours help).")


def walk(ftp, path):
    """Yield remote file paths below `path`."""
    entries = []
    ftp.retrlines(f"LIST {path}", entries.append)
    for e in entries:
        name = e.split(maxsplit=8)[-1]
        full = f"{path.rstrip('/')}/{name}"
        if e.startswith("d"):
            yield from walk(ftp, full)
        else:
            yield full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote", nargs="?", default="/")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cred", default="~/.tpdc_snow_ftp")
    a = ap.parse_args()
    ftp = connect(a.cred)
    if a.list:
        ftp.retrlines(f"LIST {a.remote}")
        ftp.quit()
        return
    files = list(walk(ftp, a.remote))
    ftp.quit()
    OUT.mkdir(parents=True, exist_ok=True)
    for remote in files:
        target = OUT / remote.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        ftp = connect(a.cred)
        ftp.voidcmd("TYPE I")
        size = ftp.size(remote)
        offset = target.stat().st_size if target.is_file() else 0
        if offset < size:
            with target.open("ab") as out:
                ftp.retrbinary("RETR " + remote, out.write, 1024 * 1024, offset if offset else None)
        ftp.quit()
        print(f"{remote}: {target.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
