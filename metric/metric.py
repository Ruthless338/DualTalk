import os
import numpy as np
import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.stats import entropy
from tqdm import tqdm
import json
pre_path = "./result_DualTalk/"
gt_path = "./data/test/"
save_name = "metric_dualtalk.json"
save_path = "./metric/"


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()




def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_variance(activations):
    return np.mean(np.var(activations, axis=0))


def calcuate_sid(gt, pred, k):
    # gt: list of [seq_len, dim]
    # pred: list of [seq_len, dim]

    # merge_gt = np.concatenate(gt, axis=0)
    merge_gt = gt
    # run kmeans on gt
    kmeans_gt = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(merge_gt)
    # run kmeans on pred

    kmeans_pred = kmeans_gt.predict(pred)
    # compute histogram
    hist_cnt = [0] * k
    for i in range(len(kmeans_pred)):
        hist_cnt[kmeans_pred[i]] += 1
    hist_cnt = np.array(hist_cnt)
    hist_cnt = hist_cnt / np.sum(hist_cnt)
    # compute entropy
    entropy = 0
    eps = 1e-6
    for i in range(k):
        entropy += hist_cnt[i] * np.log2(hist_cnt[i]+eps)
    return -entropy

def sts(x, y, timestep=0.1):
    ans = 0
    total_sample, dim = x.shape
    for di  in range(dim):
        for i in range(1, total_sample):
            ans += ((x[i][di] - x[i-1][di]) - (y[i][di] - y[i-1][di]))**2 / timestep
    return np.sqrt(ans)

fid_exp_list = []
fid_jaw_list = []
fid_pose_list = []
p_fid_exp_list = []
p_fid_jaw_list = []
p_fid_pose_list = []
mse_exp_list = []
mse_jaw_list = []
mse_pose_list = []
sid_exp_list = []
sid_jaw_list = []
sid_pose_list = []
sts_exp_list = []
sts_jaw_list = []
sts_pose_list = []
diversity_pre_exp_list = []
diversity_pre_jaw_list = []
diversity_pre_pose_list = []
diversity_gt_exp_list = []
diversity_gt_jaw_list = []
diversity_gt_pose_list = []
variance_pre_exp_list = []
variance_pre_jaw_list = []
variance_pre_pose_list = []
variance_gt_exp_list = []
variance_gt_jaw_list = []
variance_gt_pose_list = []
rpcc_exp_list = []
rpcc_jaw_list = []
rpcc_pose_list = []




for file in tqdm(os.listdir(pre_path)):
    if file.endswith(".npy"):
        pre = np.load(os.path.join(pre_path, file))
        gt = np.load(os.path.join(gt_path, file).replace(".npy", ".npz"))
        if file.endswith("speaker1.npy"):
            anthor_gt = np.load(os.path.join(gt_path, file).replace("speaker1.npy", "speaker2.npz"))
        else:
            anthor_gt = np.load(os.path.join(gt_path, file).replace("speaker2.npy", "speaker1.npz"))
        anthor_gt_exp = anthor_gt["exp"]
        anthor_gt_jaw = anthor_gt["pose"][:,3:]
        anthor_gt_pose = anthor_gt["pose"][:,:3]
        pre_exp = pre[:, :50]
        pre_jaw = pre[:, 50:53]
        pre_pose = pre[:, 53:]
        gt_exp = gt["exp"]
        gt_jaw = gt["pose"][:,3:]
        gt_pose = gt["pose"][:,:3]
        min_len = min(pre.shape[0], gt_exp.shape[0], gt_jaw.shape[0], gt_pose.shape[0], anthor_gt_exp.shape[0], anthor_gt_jaw.shape[0], anthor_gt_pose.shape[0])
        pre_exp = pre_exp[:min_len]
        pre_jaw = pre_jaw[:min_len]
        pre_pose = pre_pose[:min_len]
        gt_exp = gt_exp[:min_len]
        gt_jaw = gt_jaw[:min_len]
        gt_pose = gt_pose[:min_len]
        anthor_gt_exp = anthor_gt_exp[:min_len]
        anthor_gt_jaw = anthor_gt_jaw[:min_len]
        anthor_gt_pose = anthor_gt_pose[:min_len]

 
        # calculate frechet distance
        mu1, sigma1 = calculate_activation_statistics(pre_exp)
        mu2, sigma2 = calculate_activation_statistics(gt_exp)
        fid_exp = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        mu1, sigma1 = calculate_activation_statistics(pre_jaw)
        mu2, sigma2 = calculate_activation_statistics(gt_jaw)
        fid_jaw = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        mu1, sigma1 = calculate_activation_statistics(pre_pose)
        mu2, sigma2 = calculate_activation_statistics(gt_pose)
        fid_pose = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fid_exp_list.append(fid_exp)
        fid_jaw_list.append(fid_jaw)
        fid_pose_list.append(fid_pose)

        # calculate paired fid
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([anthor_gt_exp, gt_exp], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([anthor_gt_exp, pre_exp], axis=-1))
        fid2_exp = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([anthor_gt_jaw, gt_jaw], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([anthor_gt_jaw, pre_jaw], axis=-1))
        fid2_jaw = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([anthor_gt_pose, gt_pose], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([anthor_gt_pose, pre_pose], axis=-1))
        fid2_pose = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        p_fid_exp_list.append(fid2_exp)
        p_fid_jaw_list.append(fid2_jaw)
        p_fid_pose_list.append(fid2_pose)

        # calculate MSE
        mse_exp = np.mean((pre_exp - gt_exp)**2)
        mse_jaw = np.mean((pre_jaw - gt_jaw)**2)
        mse_pose = np.mean((pre_pose - gt_pose)**2)
        mse_exp_list.append(mse_exp)
        mse_jaw_list.append(mse_jaw)
        mse_pose_list.append(mse_pose)

        # calculate SID
        sid_exp = calcuate_sid(pre_exp, gt_exp, 40)
        sid_jaw = calcuate_sid(pre_jaw, gt_jaw, 10)
        sid_pose = calcuate_sid(pre_pose, gt_pose, 10)
        sid_exp_list.append(sid_exp)
        sid_jaw_list.append(sid_jaw)
        sid_pose_list.append(sid_pose)

        # calculate STS
        sts_exp = sts(pre_exp, gt_exp)
        sts_jaw = sts(pre_jaw, gt_jaw)
        sts_pose = sts(pre_pose, gt_pose)
        sts_exp_list.append(sts_exp)
        sts_jaw_list.append(sts_jaw)
        sts_pose_list.append(sts_pose)

        # calculate diversity
        diversity_pre_exp = calculate_diversity(pre_exp, 100)
        diversity_pre_jaw = calculate_diversity(pre_jaw, 100)
        diversity_pre_pose = calculate_diversity(pre_pose, 100)
        diversity_pre_exp_list.append(diversity_pre_exp)
        diversity_pre_jaw_list.append(diversity_pre_jaw)
        diversity_pre_pose_list.append(diversity_pre_pose)

        diversity_gt_exp = calculate_diversity(gt_exp, 100)
        diversity_gt_jaw = calculate_diversity(gt_jaw, 100)
        diversity_gt_pose = calculate_diversity(gt_pose, 100)
        diversity_gt_exp_list.append(diversity_gt_exp)
        diversity_gt_jaw_list.append(diversity_gt_jaw)
        diversity_gt_pose_list.append(diversity_gt_pose)
        # calculate variance
        variance_pre_exp = calculate_variance(pre_exp)
        variance_pre_jaw = calculate_variance(pre_jaw)
        variance_pre_pose = calculate_variance(pre_pose)
        variance_pre_exp_list.append(variance_pre_exp)
        variance_pre_jaw_list.append(variance_pre_jaw)
        variance_pre_pose_list.append(variance_pre_pose)

        variance_gt_exp = calculate_variance(gt_exp)
        variance_gt_jaw = calculate_variance(gt_jaw)
        variance_gt_pose = calculate_variance(gt_pose)
        variance_gt_exp_list.append(variance_gt_exp)
        variance_gt_jaw_list.append(variance_gt_jaw)
        variance_gt_pose_list.append(variance_gt_pose)
        # calculate rpcc

        pcc_xy_exp = np.corrcoef(gt_exp.reshape(-1, ), anthor_gt_exp.reshape(-1, ))[0, 1]
        pcc_xy_jaw = np.corrcoef(gt_jaw.reshape(-1, ), anthor_gt_jaw.reshape(-1, ))[0, 1]
        pcc_xy_pose = np.corrcoef(gt_pose.reshape(-1, ), anthor_gt_pose.reshape(-1, ))[0, 1]
        pcc_xypred_exp = np.corrcoef(pre_exp.reshape(-1, ), anthor_gt_exp.reshape(-1, ))[0, 1]
        pcc_xypred_jaw = np.corrcoef(pre_jaw.reshape(-1, ), anthor_gt_jaw.reshape(-1, ))[0, 1]
        pcc_xypred_pose = np.corrcoef(pre_pose.reshape(-1, ), anthor_gt_pose.reshape(-1, ))[0, 1]
        rpcc_exp_list.append(abs(pcc_xy_exp-pcc_xypred_exp))
        rpcc_jaw_list.append(abs(pcc_xy_jaw-pcc_xypred_jaw))
        rpcc_pose_list.append(abs(pcc_xy_pose-pcc_xypred_pose))

print("fid_exp: ", np.mean(fid_exp_list))
print("fid_jaw: ", np.mean(fid_jaw_list))
print("fid_pose: ", np.mean(fid_pose_list))
print("p_fid_exp: ", np.mean(p_fid_exp_list))
print("p_fid_jaw: ", np.mean(p_fid_jaw_list))
print("p_fid_pose: ", np.mean(p_fid_pose_list))
print("mse_exp: ", np.mean(mse_exp_list))
print("mse_jaw: ", np.mean(mse_jaw_list))
print("mse_pose: ", np.mean(mse_pose_list))
print("sid_exp: ", np.mean(sid_exp_list))
print("sid_jaw: ", np.mean(sid_jaw_list))
print("sid_pose: ", np.mean(sid_pose_list))
print("sts_exp: ", np.mean(sts_exp_list))
print("sts_jaw: ", np.mean(sts_jaw_list))
print("sts_pose: ", np.mean(sts_pose_list))
print("diversity_pre_exp: ", np.mean(diversity_pre_exp_list))
print("diversity_pre_jaw: ", np.mean(diversity_pre_jaw_list))
print("diversity_pre_pose: ", np.mean(diversity_pre_pose_list))
print("diversity_gt_exp: ", np.mean(diversity_gt_exp_list))
print("diversity_gt_jaw: ", np.mean(diversity_gt_jaw_list))
print("diversity_gt_pose: ", np.mean(diversity_gt_pose_list))
print("variance_pre_exp: ", np.mean(variance_pre_exp_list))
print("variance_pre_jaw: ", np.mean(variance_pre_jaw_list))
print("variance_pre_pose: ", np.mean(variance_pre_pose_list))
print("variance_gt_exp: ", np.mean(variance_gt_exp_list))
print("variance_gt_jaw: ", np.mean(variance_gt_jaw_list))
print("variance_gt_pose: ", np.mean(variance_gt_pose_list))
print("rpcc_exp: ", np.mean(rpcc_exp_list))
print("rpcc_jaw: ", np.mean(rpcc_jaw_list))
print("rpcc_pose: ", np.mean(rpcc_pose_list))

result = {
    "fid_exp": str(np.mean(fid_exp_list)),
    "fid_jaw": str(np.mean(fid_jaw_list)),
    "fid_pose": str(np.mean(fid_pose_list)),
    "p_fid_exp": str(np.mean(p_fid_exp_list)),
    "p_fid_jaw": str(np.mean(p_fid_jaw_list)),
    "p_fid_pose": str(np.mean(p_fid_pose_list)),
    "mse_exp": str(np.mean(mse_exp_list)),
    "mse_jaw": str(np.mean(mse_jaw_list)),
    "mse_pose": str(np.mean(mse_pose_list)),
    "sid_exp": str(np.mean(sid_exp_list)),
    "sid_jaw": str(np.mean(sid_jaw_list)),
    "sid_pose": str(np.mean(sid_pose_list)),
    "sts_exp": str(np.mean(sts_exp_list)),
    "sts_jaw": str(np.mean(sts_jaw_list)),
    "sts_pose": str(np.mean(sts_pose_list)),
    "diversity_pre_exp": str(np.mean(diversity_pre_exp_list)),
    "diversity_pre_jaw": str(np.mean(diversity_pre_jaw_list)),
    "diversity_pre_pose": str(np.mean(diversity_pre_pose_list)),
    "diversity_gt_exp": str(np.mean(diversity_gt_exp_list)),
    "diversity_gt_jaw": str(np.mean(diversity_gt_jaw_list)),
    "diversity_gt_pose": str(np.mean(diversity_gt_pose_list)),
    "variance_pre_exp": str(np.mean(variance_pre_exp_list)),
    "variance_pre_jaw": str(np.mean(variance_pre_jaw_list)),
    "variance_pre_pose": str(np.mean(variance_pre_pose_list)),
    "variance_gt_exp": str(np.mean(variance_gt_exp_list)),
    "variance_gt_jaw": str(np.mean(variance_gt_jaw_list)),
    "variance_gt_pose": str(np.mean(variance_gt_pose_list)),
    "rpcc_exp": str(np.mean(rpcc_exp_list)),
    "rpcc_jaw": str(np.mean(rpcc_jaw_list)),
    "rpcc_pose": str(np.mean(rpcc_pose_list)),
}

with open(os.path.join(save_path, save_name), "w") as f:
    json.dump(result, f)
