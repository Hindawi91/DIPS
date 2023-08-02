from fid_score import calculate_fid_given_paths
import pandas as pd
import tqdm


gpu_device = 'cpu'
dims = 192
batch_size = 1
num_workers = 0

fid_dic = {}

for i in range(10000,300001,10000):
    print(f"Getting FID for model {i}")
    paths = []
    path1 = f'./FPGAN_Training/brats_syn_256_lambda0.1/results_{i}'
    path2 = './FPGAN_Training/data/brats/syn/train/negative'
    paths.append(path1)
    paths.append(path2)

    fid_value = calculate_fid_given_paths(paths,batch_size,gpu_device,dims,num_workers=num_workers)
    model_name = str(int(i/1000))+"k"

    fid_dic[model_name] = fid_value


df = pd.DataFrame(list(fid_dic.items()),columns = ["model", "FID"])

df.to_excel (f'./FID_Result.xlsx', index = False, header=True)