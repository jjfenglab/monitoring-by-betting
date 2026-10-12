"""Loaders for CUB-200-2011 images, class names, and per-image attribute annotations.

Used by assemble_cub_tilt (attribute-weighted sampling over a fixed population)
and select_cub_tilt (the rule that picks which attributes to tilt).
"""

from pathlib import Path

import pandas as pd

# 606 of the 3.7M rows carry a stray extra field between certainty and time;
# declaring six columns keeps the first four aligned for every row, so nothing
# is dropped (the stock file needs no hand repair).
ATTRIBUTE_LABEL_COLUMNS = ["image_idx", "attribute_idx", "is_present", "certainty", "x5", "x6"]


def load_class_images(cub_dir: Path, class_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """One row per image of the given classes: observation (image path), class label, 312 binary attributes."""
    root = cub_dir / "CUB_200_2011"
    images = pd.read_csv(root / "images.txt", sep=r"\s+", header=None,
                         names=["image_idx", "image_path"])
    labels = pd.read_csv(root / "image_class_labels.txt", sep=r"\s+", header=None,
                         names=["image_idx", "class_id"])
    images = images.merge(labels, on="image_idx")
    images = images[images.class_id.isin(class_ids)]
    images["observation"] = [str(root / "images" / p) for p in images.image_path]

    attr_names = pd.read_csv(cub_dir / "attributes.txt", sep=r"\s+", header=None,
                             names=["attribute_idx", "attribute"])
    image_attrs = pd.read_csv(root / "attributes" / "image_attribute_labels.txt",
                              sep=r"\s+", header=None, names=ATTRIBUTE_LABEL_COLUMNS)
    image_attrs = image_attrs[image_attrs.image_idx.isin(images.image_idx)]
    attr_matrix = image_attrs.pivot(index="image_idx", columns="attribute_idx",
                                    values="is_present").astype(int)
    # Select by attribute_idx before renaming: naming the pivot's columns positionally
    # would silently mislabel every attribute if the two files ever diverged in order.
    attr_matrix = attr_matrix[attr_names.attribute_idx.tolist()]
    attr_matrix.columns = attr_names.attribute.tolist()

    df = images[["image_idx", "observation", "class_id"]].merge(attr_matrix, on="image_idx")
    return df.reset_index(drop=True), attr_names.attribute.tolist()


def attribute_gloss(attribute: str) -> str:
    """Render a CUB attribute name like has_wing_color::blue as a natural-language gloss."""
    part, value = attribute.removeprefix("has_").split("::")
    return f"the bird's {part.replace('_', ' ')} is {value.replace('_', ' ')}"


def class_names(cub_dir: Path, class_ids: list[int]) -> dict[int, str]:
    """Map 1-based CUB class ids to their species names."""
    classes = pd.read_csv(cub_dir / "CUB_200_2011" / "classes.txt", sep=r"\s+", header=None,
                          names=["class_id", "class_name"])
    return dict(classes[classes.class_id.isin(class_ids)].values)
