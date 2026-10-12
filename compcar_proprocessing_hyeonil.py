from pathlib import Path
import pandas as pd
import scipy.io

# ============================================================
# 1. USER INPUT
# ============================================================

# 각 팀원이 자기 CompCars 경로로 수정
path = Path(
    r"C:\Users\mgsam\OneDrive\바탕 화면\코딩\archive"
)

image_root = path / "image"
misc_root = path / "misc"

output_path = Path("compcars_metadata.csv")


# ============================================================
# 2. LOAD MAKE / MODEL NAMES
# ============================================================

mat = scipy.io.loadmat(
    misc_root / "make_model_name.mat"
)

import numpy as np
import scipy.io

mat = scipy.io.loadmat(
    misc_root / "make_model_name.mat"
)


def matlab_cell_to_string(cell):

    # MATLAB cell/array를 Python 문자열로 변환
    while isinstance(cell, np.ndarray):

        # 빈 cell
        if cell.size == 0:
            return ""

        # 첫 번째 요소를 계속 꺼냄
        cell = cell.flat[0]

    if isinstance(cell, bytes):
        cell = cell.decode("utf-8")

    return str(cell).strip()


# MATLAB 배열을 1차원으로 펼침
make_raw = mat["make_names"].ravel()
model_raw = mat["model_names"].ravel()

# ID 순서를 유지하면서 문자열 변환
make_names = [
    matlab_cell_to_string(x)
    for x in make_raw
]

model_names = [
    matlab_cell_to_string(x)
    for x in model_raw
]


print("Number of make entries :", len(make_names))
print("Number of model entries:", len(model_names))

print("\nFirst 10 make names:")
print(make_names[:10])

print("\nFirst 20 model names:")
print(model_names[:20])

print(
    "\nEmpty make entries:",
    sum(name == "" for name in make_names)
)

print(
    "Empty model entries:",
    sum(name == "" for name in model_names)
)

print("Number of make names :", len(make_names))
print("Number of model names:", len(model_names))


# ============================================================
# 3. IMAGE EXTENSIONS
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 4. CREATE METADATA
# ============================================================

records = []

for make_dir in image_root.iterdir():

    if not make_dir.is_dir():
        continue

    try:
        make_id = int(make_dir.name)
    except ValueError:
        continue

    # CompCars ID starts from 1
    make = make_names[make_id - 1]

    for model_dir in make_dir.iterdir():

        if not model_dir.is_dir():
            continue

        try:
            model_id = int(model_dir.name)
        except ValueError:
            continue

        model = model_names[model_id - 1]

        for year_dir in model_dir.iterdir():

            if not year_dir.is_dir():
                continue

            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for image_file in year_dir.iterdir():

                if (
                    image_file.is_file()
                    and image_file.suffix.lower()
                    in image_extensions
                ):

                    records.append({
                        "image_path": str(image_file),
                        "make_id": make_id,
                        "model_id": model_id,
                        "make": make,
                        "model": model,
                        "year": year
                    })


df = pd.DataFrame(records)

print("\n" + "=" * 60)
print("METADATA CREATED")
print("=" * 60)

print("Total images:", len(df))
print(df.head())


# ============================================================
# 5. NULL CHECK
# ============================================================

print("\n" + "=" * 60)
print("NULL VALUES")
print("=" * 60)

print(df.isnull().sum())

before = len(df)

df = df.dropna().copy()

print("Removed null rows:", before - len(df))


# ============================================================
# 6. CREATE MAKE-MODEL CLASS
# ============================================================

df["model_class"] = (
    df["make"].astype(str)
    + "_"
    + df["model"].astype(str)
)


# ============================================================
# 7. MODEL CLASS DISTRIBUTION
# ============================================================

model_distribution = (
    df["model_class"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("MODEL CLASS DISTRIBUTION")
print("=" * 60)

print(
    "Number of model classes :",
    len(model_distribution)
)

print(
    "Maximum images          :",
    model_distribution.max()
)

print(
    "Median images           :",
    model_distribution.median()
)

print(
    "Minimum images          :",
    model_distribution.min()
)


# ============================================================
# 8. THRESHOLD ANALYSIS
# ============================================================

thresholds = [10, 20, 30, 50, 100, 200]

print("\n" + "=" * 60)
print("MODEL CLASS THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in thresholds:

    usable = model_distribution[
        model_distribution >= threshold
    ]

    print(
        f"Images >= {threshold:3d} : "
        f"{len(usable):4d} classes"
    )


# ============================================================
# 9. SAVE
# ============================================================

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

print(
    f"Metadata saved to: "
    f"{output_path.resolve()}"
)