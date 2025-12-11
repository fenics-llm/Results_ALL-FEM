import os, shutil, stat

NAME = "oss-multi-fluid12"               # <- change once
root = "/content"
target = os.path.join(root, NAME)
os.makedirs(target, exist_ok=True)

# 1) Save previous cell's code as /content/<NAME>.py
prev_code = In[-2]  # IPython input history
with open(os.path.join(root, f"{NAME}.py"), "w", encoding="utf-8") as f:
    f.write(prev_code)

# 2) Move everything (including hidden) into /content/<NAME>, except the folder itself
skip = {NAME}  # add "drive" here if mounted and you do NOT want to touch it: {"code1","drive"}
for entry in os.listdir(root):
    if entry in skip:
        continue
    src = os.path.join(root, entry)
    dst = os.path.join(target, entry)
    try:
        shutil.move(src, dst)
    except Exception as e:
        print(f"Skipped moving {src}: {e}")

# 3) Safety sweep: remove anything that somehow still sits in /content (handles files, dirs, symlinks)
for entry in os.listdir(root):
    if entry in skip:
        continue
    path = os.path.join(root, entry)
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            def onerror(func, p, exc_info):
                try:
                    os.chmod(p, stat.S_IWRITE)
                    func(p)
                except Exception:
                    pass
            shutil.rmtree(path, onerror=onerror)
    except Exception as e:
        print(f"Could not delete {path}: {e}")
# 4) Zip the folder. Create zip in /content then move it INSIDE the folder so nothing is left outside.
zip_tmp = shutil.make_archive(os.path.join(root, NAME), "zip", root_dir=root, base_dir=NAME)
shutil.move(zip_tmp, os.path.join(target, f"{NAME}.zip"))

print(f"Done. Only /content/{NAME} remains, and {NAME}.zip is inside it.")