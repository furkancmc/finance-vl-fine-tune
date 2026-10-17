import pymupdf, re, json, sys
from pathlib import Path

NUM = re.compile(r'^\(?-?[\d]{1,3}(\.\d{3})*(,\d+)?\)?$|^-$')

def to_num(t):
    if t == '-': return 0
    neg = t.startswith('(')
    v = int(t.strip('()').replace('.', '').replace('-', '') or 0) if ',' not in t else float(t.strip('()').replace('.', '').replace(',', '.'))
    return -v if neg else v

def lines(page, tol=3):
    ws = sorted(page.get_text("words"), key=lambda w: ((w[1]+w[3])/2, w[0]))
    out, cur, cy = [], [], None
    for w in ws:
        y = (w[1]+w[3])/2
        if cy is None or abs(y-cy) <= tol: cur.append(w); cy = y if cy is None else cy
        else: out.append(sorted(cur, key=lambda w: w[0])); cur=[w]; cy=y
    if cur: out.append(sorted(cur, key=lambda w: w[0]))
    return out

def extract_table(page, ncols, x_split):
    rows, pending = [], []
    for ln in lines(page):
        text = [w[4] for w in ln if w[0] < x_split]
        nums = [w[4] for w in ln if w[0] >= x_split and NUM.match(w[4])]
        if len(nums) >= ncols:
            vals = nums[-ncols:]
            note = nums[:-ncols][0] if len(nums) > ncols else None
            label = ' '.join(pending + [' '.join(text)]).strip()
            rows.append({"kalem": label, "dipnot": note, "degerler": [to_num(v) for v in vals]})
            pending = []
        else:
            pending.append(' '.join(w[4] for w in ln))
    return rows

if __name__ == '__main__':
    pdf, pno, ncols, xs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    d = pymupdf.open(pdf); p = d[pno]
    print(json.dumps(extract_table(p, ncols, xs), ensure_ascii=False, indent=1))
