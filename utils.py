import pickle

import numpy as np
import pickle as pkl
from collections import defaultdict
import random
from sklearn.metrics.pairwise import euclidean_distances

# from diversity_baselines import MMR, DPP

def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_batch(data, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


def get_aggregated_batch(data, batch_size, batch_no):
    return [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))]


def padding_list(seq, max_len):
    seq_length = min(len(seq), max_len)
    if len(seq) < max_len:
        seq += [np.zeros_like(np.array(seq[0])).tolist()] * (max_len - len(seq))
    return seq[:max_len], seq_length


def construct_behavior_data(data, user_history, max_len, multi_hot=False):
    target_user, target_item, user_behavior, label, seq_length = [], [], [], [], []
    for d in data:
        uid, iid, cid, lb, = d
        if uid in user_history:
            target_user.append(uid)
            if multi_hot:
                ft = [iid]
                ft.extend(cid)
            else:
                ft = [iid, cid]
            target_item.append(ft)
            user_list, length = padding_list(user_history[uid], max_len)
            user_behavior.append(user_list)
            label.append(lb)
            seq_length.append(length)
    return target_user, target_item, user_behavior, label, seq_length


def get_user_div_history(user_history, max_behavior_len, items_div, multi_hot=False):
    '''
    generate separate user history for learning theta in diver
    items belong to the category that has the largest prob
    '''
    num_cat = len(items_div[user_history[0][0]])
    div_dict = {cid: [] for cid in range(num_cat)}
    user_div = []
    for iid in user_history:
        if iid[0] == 0:
            continue
        if multi_hot:
            for i, v in enumerate(iid[1:]):
                if v:
                    div_dict[i].append(iid[0])
        else:
            itm_div_class = np.argmax(np.array(items_div[iid[0]])) #items: [iid, cid]
            div_dict[itm_div_class].append(iid[0])
    for cid, div_hist in div_dict.items():
        if len(div_hist) < max_behavior_len:
            div_hist += [0] * (max_behavior_len - len(div_hist))
        user_div.extend(div_hist[:max_behavior_len])
    assert len(user_div) == num_cat * max_behavior_len
    return user_div


def dcm_theta(user_profile_dict, item_div_dir, num_clusters, user_set, out_file, multi_hot=False):
    count_cat = {uid: np.zeros(num_clusters) for uid in user_set}
    theta = {}
    items_div = pkl.load(open(item_div_dir, 'rb'))
    for uid, user_list in user_profile_dict.items():
        for itm in user_list:
            iid = itm[0]
            if iid == 0:
                continue
            if multi_hot:
                count_cat[uid] += np.array(itm[1:])
            else:
                itm_div_class = np.argmax(np.array(items_div[iid]))
                count_cat[uid][itm_div_class] += 1
    for uid, cats in count_cat.items():
        if multi_hot:
            theta[uid] = normalize(cats)
        else:
            theta[uid] = normalize(cats)
    with open(out_file, 'wb') as f:
        pkl.dump(theta, f)
    print('dcm theta saved')


def rank(users, items, preds, labels, out_file, item_div_dir, user_history, max_behavior_len, multi_hot=False):
    items_div = pkl.load(open(item_div_dir, 'rb'))
    rankings = defaultdict(list)
    with open(out_file, 'w') as fout:
        for uid, iid, pred, lb in zip(users, items, preds, labels):
            rankings[uid].append((iid, pred, lb))
        for uid, user_list in rankings.items():
            if len(user_list) >= 3:
                user_list = sorted(user_list, key=lambda x: x[1], reverse=True) # sort by predictions. moved this to load_data()
                for itm in user_list:
                    hist_u = list(map(str, get_user_div_history(user_history[uid], max_behavior_len, items_div, multi_hot)))
                    ft, p, l = itm
                    if multi_hot:
                        i, c = ft[0], list(map(int, ft[1:]))
                        div_i = list(map(str, items_div[i]))
                        fout.write(','.join([str(l), str(p), str(uid), str(i)] + list(map(str, c)) + div_i + hist_u))
                    else:
                        i, c = ft
                        div_i = list(map(str, items_div[i]))
                        fout.write(','.join([str(l), str(p), str(uid), str(i), str(c)] + div_i + hist_u))
                    fout.write('\t')
                fout.write('\n')


def get_last_click_pos(my_list):
    if sum(my_list) == 0 or sum(my_list) == len(my_list):
        return len(my_list) - 1
    return max([index for index, el in enumerate(my_list) if el])


def construct_list(data_dir, max_time_len, num_cat, is_train, multi_hot=False):
    if multi_hot:
        num_sample = 20 if is_train else 4
    else:
        num_sample = 50 if is_train else 10
    feat, click, seq_len, user_behavior, items_div, uid, pred = [], [], [], [], [], [], []
    with open(data_dir, 'r') as f:
        for line in f:
            items = line.strip().split('\t')

            uid_i, feat_i, click_i, user_i, div_i, pred_i = [], [], [], [], [], []
            for itm in items:
                itm = itm.strip().split(',')
                click_i.append(int(itm[0]))
                pred_i.append(float(itm[1]))
                uid_i.append(int(itm[2]))
                if multi_hot:
                    feat_i.append(list(map(int, list(map(float, itm[3:4 + num_cat])))))
                    div_i.append(normalize(list(map(float, itm[4 + num_cat:4 + 2*num_cat]))).tolist())
                    user_i.append(list(map(int, itm[4 + 2*num_cat:])))
                else:
                    feat_i.append(list(map(int, map(float,itm[3:5]))))
                    div_i.append(list(map(float, itm[5:5 + num_cat])))
                    user_i.append(list(map(int, itm[5 + num_cat:])))
            rankings = list(zip(click_i, pred_i, feat_i, div_i, user_i))
            if len(rankings) > max_time_len:
                for i in range(num_sample):
                    cand = random.sample(rankings, max_time_len)
                    click_i, pred_i, feat_i, div_i, user_i = zip(*cand)
                    seq_len_i = len(feat_i)
                    sorted_idx = sorted(range(len(pred_i)), key=lambda k: pred_i[k], reverse=True)
                    pred_i = np.array(pred_i)[sorted_idx].tolist()
                    click_i = np.array(click_i)[sorted_idx].tolist()
                    feat_i = np.array(feat_i)[sorted_idx].tolist()
                    div_i = np.array(div_i)[sorted_idx].tolist()

                    feat.append(feat_i[:max_time_len])
                    user_behavior.append(user_i[0])
                    click.append(click_i[:max_time_len])
                    items_div.append(div_i[:max_time_len])
                    seq_len.append(min(max_time_len, seq_len_i))
                    uid.append(uid_i[0])
                    pred.append(pred_i[:max_time_len])
            else:
                click_i, pred_i, feat_i, div_i, user_i = zip(*rankings)
                seq_len_i = len(feat_i)
                sorted_idx = sorted(range(len(pred_i)), key=lambda k: pred_i[k], reverse=True)
                pred_i = np.array(pred_i)[sorted_idx].tolist()
                click_i = np.array(click_i)[sorted_idx].tolist()
                feat_i = np.array(feat_i)[sorted_idx].tolist()
                div_i = np.array(div_i)[sorted_idx].tolist()

                feat.append(feat_i + [np.zeros_like(np.array(feat_i[0])).tolist()] * (max_time_len - seq_len_i))
                user_behavior.append(user_i[0])
                click.append(click_i + [0] * (max_time_len - seq_len_i))
                items_div.append(div_i + [np.zeros_like(np.array(div_i[0])).tolist()] * (max_time_len - seq_len_i))
                seq_len.append(seq_len_i)
                uid.append(uid_i[0])
                pred.append(pred_i + [-1e9] * (max_time_len - seq_len_i))

    return feat, click, seq_len, user_behavior, items_div, uid, pred


def load_data(data, click_model, num_cate, max_hist_len, test=False):
    feat, click, seq_len, user_behavior, items_div, uid, _ = data
    feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm = [], [], [], [], [], [], []
    i = 0
    for feat_i, click_i, seq_len_i, user_i, div_i, uid_i in zip(feat, click, seq_len, user_behavior, items_div, uid):
        behav_len = process_seq(user_i, num_cate, max_hist_len)
        item_list_i = [itm[0] for itm in feat_i]
        label_cm_i = click_model.generate_clicks(uid_i, item_list_i, len(item_list_i))
        if not test:
            if sum(label_cm_i) == len(label_cm_i) or sum(label_cm_i) == 0:
                continue

        feat_cm.append(feat_i)
        user_behavior_cm.append(user_i)
        behav_len_cm.append(behav_len)
        label_cm.append(label_cm_i)
        items_div_cm.append(div_i)
        seq_len_cm.append(seq_len_i)
        uid_cm.append(uid_i)

    return feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm


def get_hist_len(seq):
    length = 0
    while length < len(seq) and seq[length] > 0:
        length += 1
    return length

def process_seq(seq, num_cate, seq_len):
    len_list = []
    seq = np.reshape(np.array(seq), [num_cate, seq_len])
    for idx in range(num_cate):
        len_list.append(get_hist_len(seq[idx]))
    return len_list

def rerank(attracts, terms):
    val = np.array(attracts) * np.array(np.ones_like(terms))
    return sorted(range(len(val)), key=lambda k: val[k], reverse=True)


def evaluate(labels, preds, terms, users, items, click_model, items_div, scope_number, is_rerank):
    ndcg, utility, ils, diversity, satisfaction, mrr = [], [], [], [], [], []
    for label, pred, term, uid, init_list, item_div in zip(labels, preds, terms, users, items, items_div):
        init_list = np.array([item[0] for item in init_list])
        seq_len = get_last_click_pos(init_list)
        item_div = np.array(item_div)

        if is_rerank:
            # rerank list
            final = rerank(pred, term)
            final_list = init_list[final]
        else:
            final = list(range(len(pred))) # evaluate initial rankers
            final_list = init_list[final]

        click = np.array(label)[final].tolist() # reranked labels
        gold = sorted(range(len(click)), key=lambda k: click[k], reverse=True) # optimal list for ndcg
        item_div_final = item_div[final]

        ideal_dcg, dcg, util, rr = 0, 0, 0, 0
        scope_number = min(scope_number, seq_len)
        scope_final = final[:scope_number]
        scope_gold = gold[:scope_number]
        scope_div = item_div_final[:scope_number]

        for _i, _g in zip(range(1, scope_number + 1), scope_gold):
            dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, click[_g]) - 1) / (np.log2(_i + 1))
        _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.

        new_click = click_model.generate_click_prob(uid, final_list, scope_number)
        rr = 1. / (np.argmax(np.array(new_click)) + 1)

        ndcg.append(_ndcg)
        utility.append(sum(new_click))
        mrr.append(rr)
        ils.append(np.sum(euclidean_distances(scope_div, scope_div)) / (scope_number * (scope_number - 1) / 2))
        diversity.append(np.sum(1 - np.prod(1 - scope_div, axis=0)))
        satisfaction.append(click_model.generate_satisfaction(uid, final_list, scope_number))
    return np.mean(np.array(ndcg)), np.mean(np.array(utility)), np.mean(np.array(ils)), np.mean(
        np.array(diversity)), np.mean(np.array(satisfaction)), np.mean(np.array(mrr)), \
           [ndcg, utility, diversity, satisfaction, mrr]
