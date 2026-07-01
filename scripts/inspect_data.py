import numpy as np
import os

data_dir = '/home/pranav/vlsi-project/data/Vortex-small'
sample = 'Vortex-small_freq_500_mp_1_fpu_60_fpa_1.0_p_4_fi_ap.npz'

cong_base = os.path.join(data_dir, 'congestion', 'congestion_global_routing', 'overflow_based')

cell_density = np.load(os.path.join(data_dir, 'cell_density', sample))
macro_region = np.load(os.path.join(data_dir, 'macro_region', sample))
rudy = np.load(os.path.join(data_dir, 'RUDY', 'RUDY', sample))
cong_h = np.load(os.path.join(cong_base, 'congestion_GR_horizontal_overflow', sample))
cong_v = np.load(os.path.join(cong_base, 'congestion_GR_vertical_overflow', sample))

for name, arr in [('cell_density', cell_density), ('macro_region', macro_region),
                  ('rudy', rudy), ('cong_h', cong_h), ('cong_v', cong_v)]:
    key = arr.files[0]
    data = arr[key]
    print(f"{name:15s} | key={key:10s} | shape={str(data.shape):15s} | min={data.min():.3f} max={data.max():.3f} mean={data.mean():.3f}")
