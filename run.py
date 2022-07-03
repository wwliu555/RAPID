import os
import pickle
import sys
import os
import tensorflow as tf
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time
import pickle as pkl
import numpy as np

from utils import load_data, evaluate, get_aggregated_batch, construct_list
from models import PEDIR
from click_models import DCM


def eval(model, sess, data_file, max_time_len, reg_lambda, batch_size, num_cat, isrank):
    preds = []
    terms = []
    labels = []
    losses = []
    users = []
    items = []
    items_div = []

    data = load_data(data_file, click_model, test=True)
    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, term, label, loss = model.eval(sess, data_batch, reg_lambda, no_print=batch_no)
        preds.extend(pred)
        terms.extend(term)
        labels.extend(label)
        losses.append(loss)
        users.extend(data_batch[-1])
        items.extend(data_batch[0])
        items_div.extend(data_batch[4])

    if not os.path.exists('logs_{}/{}/'.format(data_set_name, max_time_len)):
        os.makedirs('logs_{}/{}/'.format(data_set_name, max_time_len))
    model_name = '{}_{}_{}_{}_{}_{}'.format(initial_rankers, model_type, batch_size, lr, reg_lambda, hidden_size)

    if isrank:
        pickle.dump([preds, users, items], open('logs_{}//{}/{}_res.pkl'.format(data_set_name, max_time_len, model_name), 'wb'))
    else:
        pickle.dump([preds, users, items], open('logs_{}//{}/{}_init.pkl'.format(data_set_name, max_time_len, model_name), 'wb'))

    loss = sum(losses) / len(losses)

    res_low = evaluate(labels, preds, terms, users, items, click_model, items_div, 5, isrank) # metrics@5
    res_high = evaluate(labels, preds, terms, users, items, click_model, items_div, 10, isrank) # metrics@10

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res_low, res_high


def train(train_file, test_file, model_type, batch_size, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len,
          item_fnum, num_cat, mu, max_norm, multi_hot):
    tf.reset_default_graph()

    if model_type == 'PEDIR':
        model = PEDIR(feature_size, eb_dim, hidden_size, max_time_len,
                      max_seq_len, item_fnum, num_cat, mu, False, max_norm, multi_hot)
    else:
        print('No Such Model')
        exit()

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'ndcg_l': [],
        'utility_l': [],
        'ils_l': [],
        'PEDIRsity_l': [],
        'satisfaction_l': [],
        'mrr_l': [],
        'ndcg_h': [],
        'utility_h': [],
        'ils_h': [],
        'PEDIRsity_h': [],
        'satisfaction_h': [],
        'mrr_h': [],
    }
    if not os.path.exists('logs_{}/{}/{}'.format(data_set_name, max_time_len, mu)):
        os.makedirs('logs_{}/{}/{}'.format(data_set_name, max_time_len, mu))
    model_name = '{}_{}_{}_{}_{}_{}'.format(initial_rankers, model_type, batch_size, lr, reg_lambda, hidden_size)
    log_save_path = 'logs_{}/{}/{}/{}.metrics'.format(data_set_name, max_time_len, mu, model_name)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, res_l, res_h = eval(model, sess, test_file, max_time_len, reg_lambda, batch_size, num_cat, False)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(None)
        training_monitor['ndcg_l'].append(res_l[0])
        training_monitor['utility_l'].append(res_l[1])
        training_monitor['ils_l'].append(res_l[2])
        training_monitor['PEDIRsity_l'].append(res_l[3])
        training_monitor['satisfaction_l'].append(res_l[4])
        training_monitor['mrr_l'].append(res_l[5])
        training_monitor['ndcg_h'].append(res_h[0])
        training_monitor['utility_h'].append(res_h[1])
        training_monitor['ils_h'].append(res_h[2])
        training_monitor['PEDIRsity_h'].append(res_h[3])
        training_monitor['satisfaction_h'].append(res_h[4])
        training_monitor['mrr_h'].append(res_h[5])

        print("STEP %d  INTIAL RANKER | LOSS VALI: NULL NDCG@L: %.4f  UTILITY@L: %.4f  ILS@L: %.4f  "
              "DIVERSE@L: %.8f SATIS@L: %.8f MRR@L: %.8f | NDCG@H: %.4f  UTILITY@H: %.4f  ILS@H: %.4f  "
              "DIVERSE@H: %.8f SATIS@H: %.8f MRR@H: %.8f" % (
        step, res_l[0], res_l[1], res_l[2], res_l[3], res_l[4], res_l[5], res_h[0], res_h[1], res_h[2], res_h[3], res_h[4], res_h[5]))
        early_stop = False

        data = load_data(train_file, click_model)
        data_size = len(data[0])
        batch_num = data_size // batch_size
        eval_iter_num = (data_size // 5) // batch_size
        print('train', data_size, batch_num)

        # begin training process
        for epoch in range(100):
            # if early_stop:
            #     break
            for batch_no in range(batch_num):
                data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
                # if early_stop:
                #     break
                loss = model.train(sess, data_batch, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, res_l, res_h = eval(model, sess, test_file, max_time_len, reg_lambda, batch_size, num_cat, True)
                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['ndcg_l'].append(res_l[0])
                    training_monitor['utility_l'].append(res_l[1])
                    training_monitor['ils_l'].append(res_l[2])
                    training_monitor['PEDIRsity_l'].append(res_l[3])
                    training_monitor['satisfaction_l'].append(res_l[4])
                    training_monitor['mrr_l'].append(res_l[5])
                    training_monitor['ndcg_h'].append(res_h[0])
                    training_monitor['utility_h'].append(res_h[1])
                    training_monitor['ils_h'].append(res_h[2])
                    training_monitor['PEDIRsity_h'].append(res_h[3])
                    training_monitor['satisfaction_h'].append(res_h[4])
                    training_monitor['mrr_h'].append(res_h[5])

                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f  NDCG@L: %.4f  UTILITY@L: %.4f  ILS@L: %.4f  "
                          "DIVERSE@L: %.8f SATIS@L: %.8f MRR@L: %.8f | NDCG@H: %.4f  UTILITY@H: %.4f  ILS@H: %.4f  "
                          "DIVERSE@H: %.8f SATIS@H: %.8f MRR@H: %.8f" % ( epoch,
                              step, train_loss, vali_loss, res_l[0], res_l[1], res_l[2], res_l[3], res_l[4],res_l[5],
                              res_h[0], res_h[1], res_h[2], res_h[3], res_h[4], res_h[5],))
                    if training_monitor['utility_l'][-1] > max(training_monitor['utility_l'][:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name)):
                            os.makedirs('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name))
                        save_path = 'save_model_{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name)
                        model.save(sess, save_path)
                        pkl.dump([res_l[-1], res_h[-1]], open(log_save_path, 'wb'))
                        print('model saved')

                    if len(training_monitor['vali_loss']) > 2 and epoch > 0:
                        if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                                training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                            early_stop = True
                        if (training_monitor['vali_loss'][-2] - training_monitor['vali_loss'][-1]) <= 0.001 and (
                                training_monitor['vali_loss'][-3] - training_monitor['vali_loss'][-2]) <= 0.001:
                            early_stop = True

        # generate log
        with open('logs_{}/{}/{}/{}.pkl'.format(data_set_name, max_time_len, mu, model_name), 'wb') as f:
            pkl.dump(training_monitor, f)


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    data_dir = 'data/'
    # data_set_name = 'taobao'
    data_set_name = 'ml-20m'
    multi_hot = True if data_set_name == 'ml-20m' else False
    stat_dir = os.path.join(data_dir, data_set_name + '/raw_data/data.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    item_div_dir = os.path.join(processed_dir, 'PEDIRsity.item')
    dcm_dir = os.path.join(data_dir, data_set_name + '/dcm.theta')
    item_fnum = 2
    user_fnum = 1
    # initial_rankers = 'svm'
    # initial_rankers = 'mart'
    initial_rankers = 'DIN'
    model_type = 'PEDIR'
    max_time_len = 20
    max_behavior_len = 5
    num_clusters = 5
    reg_lambda = 1e-4
    lr = 1e-4 
    embedding_size = 16
    batch_size = 256
    hidden_size = 32
    mu = 1.0
    max_norm = None

    user_remap_dict, item_remap_dict, cat_remap_dict, dict, feature_size = pkl.load(open(stat_dir, 'rb'))
    user_set = sorted(user_remap_dict.values())
    item_set = sorted(item_remap_dict.values())
    num_user, num_item = len(user_remap_dict), len(item_remap_dict)
    if data_set_name == 'ml-20m':
        num_clusters = len(dict)

    # construct training files
    train_dir = os.path.join(processed_dir,  initial_rankers + '.data.train')
    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        print('construct lists for training set')
        train_lists = construct_list(os.path.join(processed_dir, 'rankings.train'), max_time_len, num_clusters, True, multi_hot)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # construct test files
    test_dir = os.path.join(processed_dir, initial_rankers + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        print('construct lists for test set')
        test_lists = construct_list(os.path.join(processed_dir, 'rankings.test'), max_time_len, num_clusters, False, multi_hot)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    click_model = DCM(max_time_len, num_clusters, user_set, item_set, item_div_dir, mu, dcm_dir)
    click_model.train(train_lists)

    train(train_lists, test_lists, model_type, batch_size, feature_size, embedding_size, hidden_size, max_time_len,
          max_behavior_len, item_fnum, num_clusters, mu, max_norm, multi_hot)