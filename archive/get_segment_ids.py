import tensorstore as ts
import numpy as np

# Prevent credential issues
import os
os.environ['GCE_METADATA_ROOT'] = 'metadata.google.internal.invalid'

# Open segmentation (c3)
print("Opening segmentation...")
c3 = ts.open({
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'h01-release'},
    'path': 'data/20210601/c3',
    'scale_metadata': {'resolution': [8, 8, 33]}
}, read=True).result()[ts.d['channel'][0]]

# Take a small region (IMPORTANT: don't load whole dataset)
print("Reading small region...")
cutout = c3[320000:320512, 177000:177512, 3600].read().result()

# Get unique segment IDs
ids = np.unique(cutout)

print("\nFound segment IDs:")
print(ids[:20])  # print first 20

# Save to file
# np.savetxt("segment_ids.txt", ids, fmt="%d")
print("\nSaved all IDs to segment_ids.txt")