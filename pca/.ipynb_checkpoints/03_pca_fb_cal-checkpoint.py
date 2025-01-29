import numpy as np
import h5py
import fbpca


np.random.seed(0)


for part in np.arange(16):
    print(part)
    filename = f"/mnt/home/jzang/ceph/freq_frg/store_data/vertex_total_uddu_part{part}.h5"
    with h5py.File(filename, 'r') as file:
        vertex_uddu=np.array(file['vertex_uddu'])
    vertex_uddu=vertex_uddu.reshape(vertex_uddu.shape[0],-1)
    components =51
    print(f"component={components}")
    # Perform PCA using fbpca
    U, S, Vh = fbpca.pca(vertex_uddu, k=components, raw=True,n_iter=4,l=components+10)
    exp_var = (S**2) / np.sum(S**2)
    exp_var_ratio = exp_var * 100  # Convert to percentage
    cum_exp_var = np.cumsum(exp_var_ratio)
    print("done")
    print("Explained Variance Ratio (%):", exp_var_ratio)
    print("Cumulative Explained Variance Ratio (%):", cum_exp_var)
    filename = f"/mnt/home/jzang/ceph/freq_frg/svd/svd_all_vertex_part{part}_50.h5"
    #filename = f"/mnt/home/jzang/ceph/freq_frg/svd/train_test_split/svd_all_vertex_part{part}_50.h5"
    with h5py.File(filename, 'a') as file:
        file['U']=U
        file['S']=S
        file['Vh']=Vh

# Generate indices and shuffle them
indices = np.arange(140)
np.random.shuffle(indices)
# Split indices for training and testing
train_indices = indices[:110]
test_indices =indices[110:]# np.arange(vertex_uddu_stacked.shape[0])# indices[110:]
print(train_indices)

for part in np.arange(16):#[12,13,14,15]:
    print(part)
    filename = f"/mnt/home/jzang/ceph/freq_frg/store_data/vertex_total_uddu_part{part}.h5"
    with h5py.File(filename, 'r') as file:
        vertex_uddu=np.array(file['vertex_uddu'])
    vertex_uddu=vertex_uddu.reshape(vertex_uddu.shape[0],-1)
    vertex_uddu=vertex_uddu[train_indices]
    components =51
    print(f"component={components}")
    # Perform PCA using fbpca
    U, S, Vh = fbpca.pca(vertex_uddu, k=components, raw=True,n_iter=4,l=components+10)
    exp_var = (S**2) / np.sum(S**2)
    exp_var_ratio = exp_var * 100  # Convert to percentage

    cum_exp_var = np.cumsum(exp_var_ratio)

    print("done")
    print("Explained Variance Ratio (%):", exp_var_ratio)
    print("Cumulative Explained Variance Ratio (%):", cum_exp_var)
    # filename = f"/mnt/home/jzang/ceph/freq_frg/svd/svd_all_vertex_part{part}_50.h5"
    filename = f"/mnt/home/jzang/ceph/freq_frg/svd/train_test_split/svd_all_vertex_part{part}_50.h5"
    with h5py.File(filename, 'a') as file:
        file['U']=U
        file['S']=S
        file['Vh']=Vh
