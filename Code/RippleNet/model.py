import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class RippleNet(object):
    def __init__(self, args, num_entity, num_rel):
        self._parse_args(args, num_entity, num_rel)
        self._prepare_input()
        self._create_embeddings()
        self._create_model()
        self._compute_loss()
        self._train_model()

    def _prepare_input(self):
        self.mem_relation = []
        self.mem_head = []
        self.mem_tail = []
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")

    def _parse_args(self, args, num_entity, num_rel):
        self.num_entity = num_entity
        self.kge_weight = args.kge_weight
        self.num_rel = num_rel
        self.item_update_mode = args.item_update_mode
        self.num_memory = args.n_memory
        self.num_hops = args.n_hop
        self.dim = args.dim
        self.lr = args.lr
        self.use_all_hops = args.using_all_hops
        self.l2_weight = args.l2_weight


    def _create_embeddings(self):
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.num_rel, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.num_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())


        for hop in range(self.num_hops):
            self.mem_head.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.num_memory], name="mem_head_" + str(hop)))
            self.mem_relation.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.num_memory], name="mem_relation_" + str(hop)))
            self.mem_tail.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.num_memory], name="mem_tail_" + str(hop)))



    def _build_o_set(self):
        o_set = []
        for level in range(self.num_hops):
            h_expanded = tf.expand_dims(self.h_emb_list[level], axis=3)
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[level], h_expanded), axis=3)
            v = tf.expand_dims(self.item_embeddings, axis=2)
            probability = tf.squeeze(tf.matmul(Rh, v), axis=2)
            probability_norm = tf.nn.softmax(probability)
            extended_prob = tf.expand_dims(probability_norm, axis=2)
            o = tf.reduce_sum(self.t_emb_list[level] * extended_prob, axis=1)
            self.item_embeddings = tf.matmul(self.item_embeddings + o, self.transform_matrix) # self.update_item_embedding(self.item_embeddings, o) ###########
            o_set.append(o)
        return o_set

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)


    def _create_model(self):
        # To update item embedding
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for x in range(self.num_hops):
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.mem_head[x]))
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.mem_relation[x]))
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.mem_tail[x]))
        o_set = self._build_o_set()
        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_set))
        self.scores_normalized = tf.sigmoid(self.scores)


    def predict(self, item_embeddings, o_set):
        vector = o_set[-1]
        if self.use_all_hops:
            for i in range(self.num_hops - 1):
                vector += o_set[i]
        scores = tf.reduce_sum(item_embeddings * vector, axis=1)
        return scores

    # def update_item_embedding(self, item_embeddings, o):
    #     if self.item_update_mode == "replace":
    #         item_embeddings = o
    #     elif self.item_update_mode == "plus":
    #         item_embeddings = item_embeddings + o
    #     elif self.item_update_mode == "replace_transform":
    #         item_embeddings = tf.matmul(o, self.transform_matrix)
    #     elif self.item_update_mode == "plus_transform":
    #         item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
    #     else:
    #         raise Exception("Unknown item updating mode: " + self.item_update_mode)
    #     return item_embeddings


    def _compute_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop_level in range(self.num_hops):
            extended_h = tf.expand_dims(self.h_emb_list[hop_level], axis=2)
            extended_t = tf.expand_dims(self.t_emb_list[hop_level], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(extended_h, self.r_emb_list[hop_level]), extended_t))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for h in range(self.num_hops):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[h] * self.h_emb_list[h]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[h] * self.t_emb_list[h]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[h] * self.r_emb_list[h]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss
        self.loss =  self.kge_loss + self.base_loss + self.l2_loss



    def eval(self, sess, feed_dict, data, sample):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        if not sample:
            auc = roc_auc_score(y_true=labels, y_score=scores)
            predictions = [1 if i >= 0.5 else 0 for i in scores]
            acc = np.mean(np.equal(predictions, labels))
        else:
            scores = scores.round(decimals=3)
            # To display predcited probability list of unwatched movies of a user
            movies_file = '../data/movie/movies2.txt'
            movie_id_names = dict()
            for line in open(movies_file).readlines():
                id = line.strip().split('\t')[0]
                name = line.strip().split('\t')[1]
                movie_id_names[id] = name
            #1 printing predicted probabilities
            print("Probability prediction for user " + str(data[0][0]) + " watching the following movies")
            for m_id in movie_id_names:
                index = [i for i in range(data.shape[0]) if data[i][1] == int(m_id)]
                if not index:
                    score = 0
                    print(movie_id_names[m_id] + ": " + str(score))
                else:
                    print(movie_id_names[m_id] + ": " + str(scores[index]))
            #2 printing top-k recommendations
            sorted_indices = [index for index, value in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]
            top_k_indices = sorted_indices[:10]
            top_k_movie_id_list = [data[i][1] for i in top_k_indices]
            movies_file = '../data/movie/movies.dat'
            movie_id_names.clear()
            for line in open(movies_file).readlines():
                id = line.strip().split('::')[0]
                name = line.strip().split('::')[1]
                movie_id_names[id] = name
            print("Top-K Recommendations for user with probabilities: "+ str(data[0][0]))
            i=0
            for k_id in top_k_movie_id_list:
                print(str(i) + ": " + movie_id_names[str(k_id)] + "   " + str(scores[top_k_indices[i]]))
                i+=1
            auc=1000
            acc=1000
        return auc, acc


    def _train_model(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
