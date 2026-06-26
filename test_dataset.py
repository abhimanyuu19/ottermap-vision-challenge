from dataset import TurfDataset
from config import *

dataset = TurfDataset(
    TRAIN_IMAGES,
    TRAIN_MASKS
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print(sample["pixel_values"].shape)

print(sample["labels"].shape)