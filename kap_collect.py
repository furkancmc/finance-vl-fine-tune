"""KAP pilot veri seti toplayici.

- Ocak-Eylul 2026 bildirimlerinden tur bazli kotalarla ornekler (selection.json'a yazar, tekrar calistirinca ayni secimi kullanir).
- Her bildirim icin: ana PDF (BildirimPdf), yapisal detay JSON (gzip) ve ek PDF'ler.
- Var olan dosyalari yeniden indirmez/ustune yazmaz; yarida kalirsa kaldigi yerden devam eder.

Kullanim: python kap_collect.py
"""
import csv
import gzip
import json
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import requests

BASE = "https://www.kap.org.tr/tr/api"
ROOT = Path(__file__).parent / "data"
DOCS = ROOT / "docs"
SEL = ROOT / "selection.json"
DELAY = 0.15
WORKERS = 8
MAX_ATTACH = 5
SEED = 42

# baslik on eki -> kota (toplam 500)
QUOTAS = {
    "Finansal Rapor": 140,
    "Faaliyet Raporu": 50,
    "Özel Durum Açıklaması": 80,
    "Kar Payı Dağıtım İşlemlerine İlişkin Bildirim": 40,
    "Genel Kurul İşlemlerine İlişkin Bildirim": 25,
    "Sermaye Artırımı": 25,
    "İzahname": 20,
    "Tertip İhraç Belgesi": 10,
    "İhraç Belgesi": 10,
    "Değerleme Raporu": 20,
    "Kredi Derecelendirmesi": 15,
    "Esas Sözleşme": 15,
    "Payların Geri Alınmasına İlişkin Bildirim": 15,
    "Bağımsız Denetim Kuruluşunun Belirlenmesi": 10,
    "Yatırımcı Raporu": 10,
    "Kurumsal Yönetim Bilgi Formu": 10,
    "Pay Alım Satım Bildirimi": 5,
}
NEED_ATTACH = {"Finansal Rapor", "Faaliyet Raporu"}
MONTHS = [(f"01.{m:02d}.2026", f"{d}.{m:02d}.2026") for m, d in
          [(1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]]

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0"


def get(url, **kw):
    for attempt in range(6):
        r = s.get(url, timeout=90, **kw)
        if r.status_code != 429:
            return r
        time.sleep(10 * (attempt + 1))
    return r


def quota_key(title):
    t = title.strip()
    # daha uzun onek once eslesmeli ("Tertip İhraç Belgesi" vs "İhraç Belgesi")
    for k in sorted(QUOTAS, key=len, reverse=True):
        if t.startswith(k):
            return k
    return None


def select():
    if SEL.exists():
        return json.loads(SEL.read_text(encoding="utf-8"))
    pools = defaultdict(list)
    for f, t in MONTHS:
        r = s.post(f"{BASE}/disclosure/list/main", json={"fromDate": f, "toDate": t, "disclosureTypes": None,
                                                           "memberTypes": [], "mkkMemberOid": None}, timeout=180)
        try:
            items = r.json()
        except ValueError:
            items = None
        if not isinstance(items, list):
            print(f"UYARI: {f}-{t} alinamadi: {r.text[:100]}")
            continue
        print(f"{f}-{t}: {len(items)} bildirim")
        for it in items:
            b = it["disclosureBasic"]
            k = quota_key(b["title"])
            if not k or b.get("isBlocked"):
                continue
            if k in NEED_ATTACH and not b.get("attachmentCount"):
                continue
            pools[k].append({"disclosureIndex": b["disclosureIndex"], "title": b["title"], "type": k,
                             "company": b["companyTitle"], "stockCode": b["stockCode"],
                             "publishDate": b["publishDate"], "attachmentCount": b.get("attachmentCount")})
        time.sleep(DELAY)
    rnd = random.Random(SEED)
    chosen = []
    for k, q in QUOTAS.items():
        pool = pools[k]
        # sirket basina cesitlilik: once her sirketten bir tane
        rnd.shuffle(pool)
        seen, first, rest = set(), [], []
        for p in pool:
            (rest if p["company"] in seen else first).append(p)
            seen.add(p["company"])
        pick = (first + rest)[:q]
        print(f"{k}: havuz {len(pool)}, secilen {len(pick)}/{q}")
        chosen += pick
    SEL.parent.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    return chosen


def unwrap_java_bytes(data):
    """api/file/download Java serialize edilmis byte[] doner: ac ed 00 05 ... 78 70 <int32 uzunluk> <veri>."""
    if data[:4] != b"\xac\xed\x00\x05":
        return data
    i = data.find(b"xp")
    n = struct.unpack(">I", data[i + 2:i + 6])[0]
    return data[i + 6:i + 6 + n]


def collect(it):
    idx = it["disclosureIndex"]
    d = DOCS / str(idx)
    d.mkdir(parents=True, exist_ok=True)
    status = {}

    detail_path = d / "detail.json.gz"
    if detail_path.exists():
        detail = json.loads(gzip.open(detail_path, "rt", encoding="utf-8").read())
    else:
        r = get(f"{BASE}/notification/attachment-detail/{idx}")
        time.sleep(DELAY)
        if not r.ok:
            return {"detail": f"fail:{r.status_code}"}
        detail = r.json()[0] if isinstance(r.json(), list) else r.json()
        with gzip.open(detail_path, "wt", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False)
    status["detail"] = "ok"

    main = d / "main.pdf"
    if main.exists():
        status["main"] = "exists"
    else:
        r = get(f"{BASE}/BildirimPdf/{idx}")
        time.sleep(DELAY)
        if r.ok and r.content[:4] == b"%PDF":
            main.write_bytes(r.content)
            status["main"] = "ok"
        else:
            status["main"] = f"fail:{r.status_code}"

    atts = [a for a in (detail.get("attachments") or []) if (a.get("fileExtension") or "").lower() == "pdf"]
    for n, a in enumerate(atts[:MAX_ATTACH], 1):
        p = d / f"attachment_{n}.pdf"
        if p.exists():
            status[f"att{n}"] = "exists"
            continue
        r = get(f"{BASE}/file/download/{a['objId']}")
        time.sleep(DELAY)
        if not r.ok:
            status[f"att{n}"] = f"fail:{r.status_code}"
            continue
        data = unwrap_java_bytes(r.content)
        if data[:4] != b"%PDF":
            status[f"att{n}"] = "fail:notpdf"
            continue
        p.write_bytes(data)
        status[f"att{n}"] = "ok"
    it["attachments"] = [{"n": n, "fileName": a.get("fileName"), "objId": a["objId"]}
                         for n, a in enumerate(atts[:MAX_ATTACH], 1)]
    return status


def main():
    ROOT.mkdir(exist_ok=True)
    items = select()
    print(f"toplam secilen: {len(items)}")
    log = {}
    lock = threading.Lock()
    done = 0

    def work(i, it):
        try:
            return i, it, collect(it)
        except Exception as e:
            return i, it, {"error": f"fail:{type(e).__name__}"}

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, i, it) for i, it in enumerate(items)]
        for fu in as_completed(futs):
            i, it, st = fu.result()
            bad = [k for k, v in st.items() if v.startswith("fail")]
            with lock:
                done += 1
                print(f"[{done}/{len(items)}] {it['disclosureIndex']} {it['type'][:25]} {'HATA ' + str(bad) if bad else 'ok'}", flush=True)
                log[i] = {**{k: v for k, v in it.items() if k != 'attachments'},
                          "n_attachments": len(it.get("attachments", [])), "status": json.dumps(st)}
                rows = [log[k] for k in sorted(log)]
                tmp = ROOT / "metadata.csv.tmp"
                with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                tmp.replace(ROOT / "metadata.csv")


if __name__ == "__main__":
    main()
