
import numpy as np
from scipy import sparse as sp
from ind_model import ind_model as model
import argparse
import cPickle

DATASET = 'citeseer'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 200)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 20)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 20)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-3)
parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'ind.model')
parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)
parser.add_argument('--fixed_train_size', type = int, default=600)
args = parser.parse_args()


def rebalance_train_test(x, y, tx, ty, allx, graph, test_indices, fixed_train_size):
    x = x.todense()
    points_to_add = fixed_train_size - len(x)
    tx = tx.todense()
    allx = allx.todense()
    new_allx = np.zeros((allx.shape[0] + points_to_add, allx.shape[1]), dtype=np.int32)
    new_x = np.zeros((fixed_train_size, x.shape[1]), dtype=np.int32)
    new_y = np.zeros((fixed_train_size, y.shape[1]), dtype=np.int32)
    new_tx = tx[points_to_add:]
    new_ty = ty[points_to_add:]

    new_allx[:len(x), :] = x
    new_allx[len(x):fixed_train_size, :] = tx[:points_to_add]
    new_allx[fixed_train_size:, :] = allx[len(x):]
    new_x[:len(x), :] = x
    new_x[len(x):, :] = tx[:points_to_add]
    new_y[:len(y), :] = y
    new_y[len(y):, :] = ty[:points_to_add]

    index_map = {}
    for i in range(len(x)):
        index_map[i] = i
    for i in range(points_to_add):
        index_map[test_indices[i]] = i + len(x)
    for i in range(fixed_train_size, len(allx)):
        index_map[i - points_to_add] = i

    new_graph = {}
    for k, v in graph.items():
        new_graph[index_map.get(k, k)] = []
        for vv in v:
            new_graph[index_map.get(k, k)].append(index_map.get(vv, vv))

    # Map everything back to corresponding sparse/np types
    return sp.csr_matrix(new_x, dtype=np.float32), new_y, sp.csr_matrix(new_tx, dtype=np.float32), new_ty, sp.csr_matrix(allx, dtype=np.float32), graph

def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

# load the data: x, y, tx, ty, allx, graph
NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
OBJECTS = []
for i in range(len(NAMES)):
    OBJECTS.append(cPickle.load(open("data/ind.{}.{}".format(DATASET, NAMES[i]))))

test_indices = open("data/ind.{}.{}".format(DATASET, "test.index")).read().split("\n")
test_indices = [int(i) for i in test_indices if len(i) > 0]


x, y, tx, ty, allx, graph = tuple(OBJECTS)
x, y, tx, ty, allx, graph = rebalance_train_test(x, y, tx, ty, allx, graph, test_indices, args.fixed_train_size)

m = model(args)                                                 # initialize the model
m.add_data(x, y, allx, graph)                                   # add data
m.build()                                                       # build the model
m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
iter_cnt, max_accu = 0, 0
patience = 50000
patience_counter = 0
while True:
    m.step_train(max_iter = 1, iter_graph = 0.1, iter_inst = 1, iter_label = 0) # perform a training step
    tpy = m.predict(tx)                                                         # predict the dev set
    accu = comp_accu(tpy, ty)                                                   # compute the accuracy on the dev set
    if iter_cnt % 1000:
        print iter_cnt, accu, max_accu
    iter_cnt += 1
    if accu > max_accu:
        m.store_params()                                                        # store the model if better result is obtained
        max_accu = max(max_accu, accu)
        patience_counter = 0
    else:
        patience_counter += 1