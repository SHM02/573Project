import numpy as np
import os
import collections


def load_rating(args):
    print('Reading ratings file...')
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_matrix = np.load(rating_file + '.npy')
    else:
        rating_matrix = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_matrix)
    return split_data(rating_matrix)

def get_ripple_set(args, kg, user_hist_dictionary):
    print('Ripple set building in progress...')
    r_set = collections.defaultdict(list)
# Iterate through user dictionary for every user
    for u in user_hist_dictionary:
        for hop in range(args.n_hop):
            mem_head = []
            mem_relation = []
            mem_tail = []

            if hop == 0:
                tails_of_previous_hop = user_hist_dictionary[u]
            else:
                tails_of_previous_hop = r_set[u][-1][2]

            for e in tails_of_previous_hop:
                for tail_and_relation in kg[e]:
                    mem_head.append(e)
                    mem_relation.append(tail_and_relation[1])
                    mem_tail.append(tail_and_relation[0])
            if len(mem_head) == 0:
                r_set[u].append(r_set[u][-1])
            else:
                flag = len(mem_head) < args.n_memory
                indexlist = np.random.choice(len(mem_head), size=args.n_memory, replace=flag)
                mem_head = [mem_head[i] for i in indexlist]
                mem_relation = [mem_relation[i] for i in indexlist]
                mem_tail = [mem_tail[i] for i in indexlist]
                r_set[u].append((mem_head, mem_relation, mem_tail))

    return r_set

def load_data(args):
    train_data, test_data, user_hist_dictionary, sample_data = load_rating(args)
    num_entity, num_rel, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_hist_dictionary)
    return train_data, test_data, num_entity, num_rel, ripple_set, sample_data

def load_kg(args):
    print('Loading Knowledge Graph...')
    # reading the KG file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_val = np.load(kg_file + '.npy')
    else:
        kg_val = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_val)
    num_entity = len(set(kg_val[:, 0]) | set(kg_val[:, 2]))
    num_rel = len(set(kg_val[:, 1]))
    kg = construct_kg(kg_val)
    return num_entity, num_rel, kg


def split_data(rating_ar):
    print('splitting dataset based on test train ratio (2:8)...')
    test_ratio = 0.2
    n_r = rating_ar.shape[0]
    test_index = np.random.choice(n_r, size=int(n_r * test_ratio), replace=False)
    left = set(range(n_r)) - set(test_index)
    train_index = list(left - set(test_index))
    user_history = dict()
    for i in train_index:
        user = rating_ar[i][0]
        item = rating_ar[i][1]
        rating = rating_ar[i][2]
        if rating == 1:
            if user not in user_history:
                user_history[user] = []
            user_history[user].append(item)

    train_i = [i for i in train_index if rating_ar[i][0] in user_history]
    test_i = [i for i in test_index if rating_ar[i][0] in user_history]

    train_data = rating_ar[train_i]
    test_data = rating_ar[test_i]
    sample_user = test_data[0][0]
    sample_data = []
    m_file = '../data/movie/movies.dat'
    all_movie_list =  []
    for line in open(m_file).readlines():
        all_movie_list.append(line.strip().split('::')[0])
    ratings_master_file = '../data/movie/ratings.dat'
    watched_movie_id_list = []
    for line in open(ratings_master_file).readlines():
        if line.strip().split('::')[0] == str(sample_user):
            watched_movie_id_list.append(line.strip().split('::')[1])
    unwatched_list = [id for id in all_movie_list if id not in watched_movie_id_list]
    for id in unwatched_list:
        sample_data.append([sample_user, int(id), 0])
    sample_data_np = np.array(sample_data)

    print("chosen user ID:")
    print(sample_user)
    print("watched movie IDs list:")
    print(watched_movie_id_list)
    print("count:" + str(len(watched_movie_id_list)))
    print("unwatched movie IDs list:")
    print(unwatched_list)
    print("count:" + str(len(unwatched_list)))
    return train_data, test_data, user_history, sample_data_np

def construct_kg(kg_ar):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for h, r, t in kg_ar:
        kg[h].append((t, r))
    return kg