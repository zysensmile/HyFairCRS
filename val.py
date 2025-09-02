import pprint
import math
import numpy as np
from numba import njit, prange

models = [
    "DCHL",
    "MHIM",
]
datasets = [
    # "OpenDialKG",
    # "DuRecDial",
    "HReDial",
    "HTGReDial",
]
epochs = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

cov_num = {
    "HReDial": 13597,
    "HTGReDial": 24914,
}

def cal_APR(data, k):
    total = []
    n = len(data) // 50
    for idx in range(n):
        total += data[idx * 50:idx * 50 + k]
    res = len(total) / (len(set(total)) * n)
    return res

def cal_LTR(data, k, p):
    total = {}
    n = len(data) // 50
    for idx in range(n):
        for rank in data[idx * 50:idx * 50 + k]:
            total[rank] = total.get(rank, 0) + 1
    tot, res = 0, 0
    for rank, num in total.items():
        tot += 1
        if num >= p * n:
            res += 1
    res = res / tot
    return res

def cal_Cov(data, k, dataset):
    total = []
    n = len(data) // 50
    for idx in range(n):
        total += data[idx * 50:idx * 50 + k]
    res = len(set(total))
    res = res / cov_num[dataset]
    return res

@njit(parallel=True)
def cal_Gini_njit(n, freq):
    res = 0
    for i in prange(n):
        for j in range(n):
            res += abs(freq[i] - freq[j])
    return res

def cal_Gini(data, k):
    n        = len(data) // 50
    top_k    = []
    for idx in range(n):
        top_k += data[idx * 50:idx * 50 + k]

    freq_dict = {}
    for item in top_k:
        freq_dict[item] = freq_dict.get(item, 0) + 1

    n        = len(freq_dict)
    avg_freq = sum(freq_dict.values()) / n
    freq     = np.array(list(freq_dict.values()), dtype=np.float64)
    gini     = cal_Gini_njit(n, freq)
    gini    /= (2 * n ** 2 * avg_freq)
    return gini

"""
计算KL散度
- 越小越好
- 简化：考虑Fair distribution(d2)是一个均匀分布
- 简化：用物品推荐次数计算Item distribution，实际应该是根据历史数据、用户行为得到的
"""
def cal_KL_divergence(data, k):
    n        = len(data) // 50
    top_k    = []
    for idx in range(n):
        top_k += data[idx * 50:idx * 50 + k]

    d1       = {}
    for item in top_k:
        d1[item] = d1.get(item, 0) + 1
    for item in d1:
        d1[item] /= len(top_k)

    d2       = {item: 1 / len(top_k) for item in set(top_k)}
    kl_div   = 0
    for item in d1:
        if d1[item] > 0 and d2.get(item, 0) > 0:
            kl_div += d1[item] * math.log(d1[item] / d2[item])
    return kl_div

@njit(parallel=True)
def cal_Difference_njit(data, pred_scores, threshold):
    n          = len(data)
    diff_count = 0
    for i in prange(n):
        for j in range(i + 1, n):
            if abs(pred_scores[i] - pred_scores[j]) < threshold:
                diff_count += 1
    return diff_count / (n * (n - 1) / 2)

"""
计算Difference
- 越大越好
- 简化：使用索引的倒数作为分数
- 阈值可随意设置
"""
def cal_Difference(data, k, threshold=0.1):
    n        = len(data) // 50
    top_k    = []
    for idx in range(n):
        top_k += data[idx * 50:idx * 50 + k]

    unique_items    = list(set(top_k))
    pred_scores_dict = {item: 1 / (index + 1) for index, item in enumerate(unique_items)}
    pred_scores     = np.array([pred_scores_dict[item] for item in top_k], dtype=np.float64)

    return cal_Difference_njit(np.array(top_k), pred_scores, threshold)

res = {}

for model in models:
    res[model] = {}
    for dataset in datasets:
        res[model][dataset] = {}
        for epoch in epochs:
            res[model][dataset][epoch] = {}
            log_path = f"rank/{dataset}/-{epoch}"
            data     = [int(num) for num in open(log_path, "r", encoding="utf-8").readline().split(" ")[:-1]]
            for k in [5, 10, 15, 20]:
                apr  = cal_APR(data, k)
                res[model][dataset][epoch][f"APR@{k}"] = f"{apr:.4f}"
                ltr  = cal_LTR(data, k, apr)
                res[model][dataset][epoch][f"LTR@{k}"] = f"{ltr:.4f}"
                cov  = cal_Cov(data, k, dataset)
                res[model][dataset][epoch][f"Cov@{k}"] = f"{cov:.4f}"
                gini = cal_Gini(data, k)
                res[model][dataset][epoch][f"Gini@{k}"] = f"{gini:.4f}"
                diff = cal_Difference(data, k, threshold=0.001)
                res[model][dataset][epoch][f"Diff@{k}"] = f"{diff:.4f}"
                kl   = cal_KL_divergence(data, k)
                res[model][dataset][epoch][f"KL@{k}"]   = f"{kl:.4f}"

print(
    "|                         | APR@5  | APR@10 | APR@15 | APR@20 | LTR@5  | LTR@10 | LTR@15 | LTR@20 | C@5    | C@10   | C@15   | C@20   |"
)
print(
    "| ----------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |"
)
for dataset in datasets:
    for model in models:
        for epoch in epochs:
            info1 = f"| {dataset:8s} - {model:8s} - {epoch}|"
            for key in [
                "APR@5", "APR@10", "APR@15", "APR@20",
                "LTR@5", "LTR@10", "LTR@15", "LTR@20",
                "Cov@5", "Cov@10", "Cov@15", "Cov@20",
            ]:
                info1 += f" {res[model][dataset][epoch][key]} |"
            print(info1)

print("")

print(
    "|                         | Gini@5  | Gini@10 | Gini@15 | Gini@20 | KL@5    | KL@10   | KL@15   | KL@20   | Diff@5  | Diff@10 | Diff@15 | Diff@20 |"
)
print(
    "| ----------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |"
)
for dataset in datasets:
    for model in models:
        for epoch in epochs:
            info2 = f"| {dataset:8s} - {model:8s} - {epoch}|"
            for key in [
                "Gini@5", "Gini@10", "Gini@15", "Gini@20",
                "KL@5", "KL@10", "KL@15", "KL@20",
                "Diff@5", "Diff@10", "Diff@15", "Diff@20",
            ]:
                info2 += f" {res[model][dataset][epoch][key]}  |"
            print(info2)
