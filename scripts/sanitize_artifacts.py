import os, re, sys
ROOTS = ["artifacts", "metrics"]
BAD = re.compile(r'[":<>|*?\r\n/\\]')  # недопустимые для upload-artifact символы
renamed = 0
for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for dirpath, _, files in os.walk(root):
        for f in files:
            if BAD.search(f):
                nf = BAD.sub("_", f)
                src = os.path.join(dirpath, f)
                dst = os.path.join(dirpath, nf)
                if src != dst:
                    os.rename(src, dst)
                    renamed += 1
print(f"Renamed {renamed} files")
