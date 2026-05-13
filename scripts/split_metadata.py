import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split metadata by identity/video groups.")
    parser.add_argument("--metadata", default="data/processed_metadata.csv")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--val", default="data/val.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def group_key(frame: pd.DataFrame) -> pd.Series:
    identity = frame.get("identity_id", pd.Series([""] * len(frame))).fillna("").astype(str)
    video = frame.get("video_id", pd.Series([""] * len(frame))).fillna("").astype(str)
    source = frame.get("source", pd.Series([""] * len(frame))).fillna("").astype(str)
    fallback = frame["path"].astype(str)
    return (identity + "::" + video + "::" + source).where((identity + video).str.len() > 0, fallback)


def write_csv(frame: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.metadata)
    groups = group_key(frame)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(splitter.split(frame, frame["label"], groups))
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    val_fraction_of_remaining = args.val_size / max(1e-6, 1.0 - args.test_size)
    groups_train_val = group_key(train_val)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_remaining, random_state=args.seed + 1)
    train_idx, val_idx = next(splitter.split(train_val, train_val["label"], groups_train_val))

    write_csv(train_val.iloc[train_idx].reset_index(drop=True), args.train)
    write_csv(train_val.iloc[val_idx].reset_index(drop=True), args.val)
    write_csv(test, args.test)


if __name__ == "__main__":
    main()
