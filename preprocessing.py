from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. USER INPUT
# ============================================================

# 각 팀원이 자기 노트북의 VMMRdb 경로만 수정
path = Path(
    r"C:\Users\mgsam\OneDrive\바탕 화면\코딩\VMMRdb"
)

# metadata 저장 위치
output_path = Path("vmmrdb_metadata.csv")


# ============================================================
# 2. IMAGE EXTENSIONS
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 3. CREATE METADATA
# ============================================================

records = []

for folder in path.iterdir():

    if not folder.is_dir():
        continue

    folder_name = folder.name

    # 예:
    # acura_integra_1994
    parts = folder_name.split("_")

    # 최소 make_model_year 구조가 아니면 skip
    if len(parts) < 3:
        print(f"Skipped folder: {folder_name}")
        continue

    # 마지막 값은 year
    year_text = parts[-1]

    try:
        year = int(year_text)
    except ValueError:
        print(f"Invalid year: {folder_name}")
        continue

    # 현재 구조에서는 첫 단어를 make로 사용
    make = parts[0]

    # 중간 부분을 model로 사용
    model = "_".join(parts[1:-1])

    # 폴더 내부 이미지 탐색
    for image_file in folder.iterdir():

        if (
            image_file.is_file()
            and image_file.suffix.lower() in image_extensions
        ):

            records.append({
                "image_path": str(image_file),
                "class_name": folder_name,
                "make": make,
                "model": model,
                "year": year
            })


# DataFrame 생성
df = pd.DataFrame(records)

print("=" * 60)
print("METADATA CREATED")
print("=" * 60)

print(f"Total images : {len(df)}")
print(f"Columns      : {list(df.columns)}")

print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# 4. NULL VALUES
# ============================================================

print("\n" + "=" * 60)
print("NULL VALUES")
print("=" * 60)

print(df.isnull().sum())

before_null = len(df)

df = df.dropna().copy()

after_null = len(df)

print(f"\nRows before : {before_null}")
print(f"Rows after  : {after_null}")
print(f"Removed     : {before_null - after_null}")


# ============================================================
# 5. COLUMN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("COLUMN DISTRIBUTION")
print("=" * 60)

for col in ["make", "model", "year", "class_name"]:

    print(f"\n[{col}]")

    print(
        "Number of unique classes :",
        df[col].nunique()
    )

    if pd.api.types.is_numeric_dtype(df[col]):

        print("Minimum :", df[col].min())
        print("Maximum :", df[col].max())
        print("Median  :", df[col].median())

    else:

        counts = df[col].value_counts()

        print(
            "Maximum class count :",
            counts.max()
        )

        print(
            "Median class count  :",
            counts.median()
        )

        print(
            "Minimum class count :",
            counts.min()
        )


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    df["class_name"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(class_distribution)

print(
    "\nNumber of classes :",
    len(class_distribution)
)

print(
    "Maximum images per class :",
    class_distribution.max()
)

print(
    "Median images per class :",
    class_distribution.median()
)

print(
    "Minimum images per class :",
    class_distribution.min()
)


# ============================================================
# 7. DETECT UNUSUAL CLASSES
# ============================================================

Q1 = class_distribution.quantile(0.25)
Q3 = class_distribution.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

unusual_classes = class_distribution[
    (class_distribution < lower_bound)
    | (class_distribution > upper_bound)
]

print("\n" + "=" * 60)
print("UNUSUAL CLASS SIZES")
print("=" * 60)

print(f"Q1          : {Q1}")
print(f"Q3          : {Q3}")
print(f"IQR         : {IQR}")
print(f"Lower bound : {lower_bound}")
print(f"Upper bound : {upper_bound}")

print("\nUnusual classes:")
print(unusual_classes)


# ============================================================
# 8. YEAR RANGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("YEAR RANGE CHECK")
print("=" * 60)

invalid_year = df[
    (df["year"] < 1950)
    | (df["year"] > 2016)
]

print(
    "Invalid year rows :",
    len(invalid_year)
)

if len(invalid_year) > 0:
    print(invalid_year)

# ============================================================
# MODEL CLASS (MAKE + MODEL, YEAR IGNORED)
# ============================================================

df["model_class"] = (
    df["make"].astype(str)
    + "_"
    + df["model"].astype(str)
)

model_distribution = (
    df["model_class"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("MODEL CLASS DISTRIBUTION")
print("=" * 60)

print(f"Number of model classes : {len(model_distribution)}")
print(f"Maximum images          : {model_distribution.max()}")
print(f"Median images           : {model_distribution.median()}")
print(f"Minimum images          : {model_distribution.min()}")

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
# 9. SAVE METADATA
# ============================================================

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

print(
    f"Metadata saved to: {output_path.resolve()}"
)