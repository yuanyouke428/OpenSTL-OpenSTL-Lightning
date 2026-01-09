import h5py

f = h5py.File('data/sevir/sevir.h5', 'r')
print("Top level keys (Groups):", list(f.keys()))
# 预期输出: ['test', 'train', 'val']

print("Train shape:", f['train']['data'].shape)
print("Val shape:  ", f['val']['data'].shape)
print("Test shape: ", f['test']['data'].shape)
# 预期形状: (N, 25, 384, 384)