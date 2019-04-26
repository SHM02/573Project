import tensorflow as tf
import numpy as np
from model import RippleNet
import matplotlib.pyplot as plt1


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_d = dict()
    feed_d[model.items] = data[start:end, 1]
    feed_d[model.labels] = data[start:end, 2]
    for ind in range(args.n_hop):
        feed_d[model.mem_head[ind]] = [ripple_set[usr][ind][0] for usr in data[start:end, 0]]
        feed_d[model.mem_relation[ind]] = [ripple_set[usr][ind][1] for usr in data[start:end, 0]]
        feed_d[model.mem_tail[ind]] = [ripple_set[user][ind][2] for user in data[start:end, 0]]
    return feed_d


def train(args, data, show_loss):
    sample_data = data[5]
    train_set = data[0]
    test_set = data[1]
    num_entity = data[2]
    ripple_set = data[4]
    num_rel = data[3]

    model = RippleNet(args, num_entity, num_rel)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_accs = []
        test_accs = []
        for s in range(args.n_epoch):
            np.random.shuffle(train_set)
            st = 0
            while st < train_set.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_set, ripple_set, st, st + args.batch_size))
                st += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (st / train_set.shape[0] * 100, loss))

            # evaluation
            sample = True
            train_auc, train_acc = evaluation(sess, args, model, train_set, ripple_set, args.batch_size, not sample)
            test_auc, test_acc = evaluation(sess, args, model, test_set, ripple_set, args.batch_size, not sample)
            sample_auc, sample_acc = evaluation(sess, args, model, sample_data, ripple_set, sample_data.shape[0], sample)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print('epoch %d    train auc: %.4f  acc: %.4f  test auc: %.4f  acc: %.4f'
                  % (s, train_auc, train_acc, test_auc, test_acc))
        x_axis = []
        for i in range(len(train_accs)):
            x_axis.append(i)
        axes = plt1.gca()
        axes.set_ylim([0, 1])
        plt1.plot(x_axis, train_accs, 'r')
        plt1.plot(x_axis, test_accs, 'g')
        plt1.savefig('test2png.png', dpi=100)


def evaluation(sess, args, model, data, ripple_set, batch_size, sample):
    s = 0
    auc_list = []
    accuracy_list = []
    while s < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, s, s + batch_size),data, sample)
        auc_list.append(auc)
        accuracy_list.append(acc)
        s += batch_size
    return float(np.mean(auc_list)), float(np.mean(accuracy_list))
