from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from livestock_dx.detect.dataset import (DEFAULT_CLASSES, build_dataset, clean_label_file, parse_label_line,
                                         polygon_to_box, split_of)


def noisy(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(path)


def make_src(root: Path):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(12):
        noisy(root / "images" / f"a{i}.jpg", i)
        (root / "labels" / f"a{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2\n")
    (root / "images" / "dup.jpg").write_bytes((root / "images" / "a0.jpg").read_bytes())          # exact duplicate
    (root / "labels" / "dup.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "images" / "bad.jpg").write_bytes(b"not an image")
    noisy(root / "images" / "multi.jpg", 99)
    (root / "labels" / "multi.txt").write_text("0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n1 1.5 0.5 0.2 0.2\n"
                                               "1 0.2 0.2 0.8 0.2 0.5 0.9\nabc\n")                   # ok, bad id, >1, polygon, junk
    noisy(root / "images" / "bg.jpg", 77)                                                            # no label file = background


def test_parse_label_line():
    assert parse_label_line("1 0.5 0.5 0.2 0.3", 4) == (1, [0.5, 0.5, 0.2, 0.3])
    for bad in ("4 0.5 0.5 0.2 0.3", "0 0.5 0.5 0 0.3", "0 1.2 0.5 0.2 0.2", "0 0.5 0.5", "x 0.5 0.5 0.2 0.2",
                "0 0.1 0.1 0.2 0.2 0.3"):                       # id out of range, zero size, >1, short, non-numeric, odd count
        with pytest.raises(ValueError):
            parse_label_line(bad, 4)
    assert len(parse_label_line("0 0.1 0.1 0.9 0.1 0.5 0.9", 4)[1]) == 6        # polygon


def test_polygon_to_box():
    assert polygon_to_box([0.2, 0.2, 0.8, 0.2, 0.5, 0.9]) == pytest.approx([0.5, 0.55, 0.6, 0.7])


def test_detect_vs_segment_label_cleaning(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.8 0.2 0.5 0.9\n")
    good, errs = clean_label_file(f, 4, "detect")
    assert len(good) == 2 and not errs and all(len(g.split()) == 5 for g in good)        # polygon became a box
    good, errs = clean_label_file(f, 4, "segment")
    assert len(good) == 1 and len(errs) == 1 and len(good[0].split()) == 7              # plain box rejected


def test_build_dataset(tmp_path):
    make_src(tmp_path / "src")
    st = build_dataset([tmp_path / "src"], tmp_path / "out", DEFAULT_CLASSES, "detect")
    assert st["dropped_unreadable_image"] == 1 and st["dropped_duplicate_image"] == 1
    assert st["dropped_bad_label_lines"] == 3                      # bad class id, out-of-range coordinate, junk
    assert st["kept_background"] == 1                              # image without labels kept as a background example
    assert st["kept_train"] + st["kept_val"] == 14
    assert st["instances_per_class"]["lumpy_skin_nodules"] >= 6 and sum(st["instances_per_class"].values()) == 14
    cfg = yaml.safe_load((tmp_path / "out" / "dataset.yaml").read_text())
    assert cfg["names"] == dict(enumerate(DEFAULT_CLASSES)) and cfg["train"] == "images/train"
    for sp in ("train", "val"):                                   # every image has a label file, and none are shared
        imgs = {p.stem for p in (tmp_path / "out" / "images" / sp).iterdir()}
        assert imgs == {p.stem for p in (tmp_path / "out" / "labels" / sp).iterdir()}
    tr = {p.stem for p in (tmp_path / "out" / "images" / "train").iterdir()}
    va = {p.stem for p in (tmp_path / "out" / "images" / "val").iterdir()}
    assert not tr & va


def test_split_is_deterministic():
    assert split_of("abc") == split_of("abc") and {split_of(str(i)) for i in range(200)} == {"train", "val"}
