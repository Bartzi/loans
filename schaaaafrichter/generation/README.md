# Scripts to generate training data

First define bounding boxes for a set of background images (see the [data folder](../data) for how to get background images):
```
python generation/annotate.py data/background/*.jpg
```

Then paste other images into the defined bounding boxes:
```
python generation/generate.py \
  --test-stamps generation/stamps/test*.png \
  --train-stamps generation/stamps/train*.png \
  --image-folder data/backgrounds \
  --search-path data/bounding_boxes
```

Alternatively just use pregenerated data (see the [data folder](../data)).
