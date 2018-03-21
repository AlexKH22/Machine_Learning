import numpy as np
from scipy.sparse import lil_matrix
from scipy import stats
import torch
import tsv
from torch.autograd import Variable
import pandas as pd


def get_datasets(num):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    
    train = pd.read_csv('ml-10M100K/r{}.train'.format(num), sep='::', names=names, engine='python')
    test = pd.read_csv('ml-10M100K/r{}.test'.format(num), sep='::', names=names, engine='python')

    return train, test


def get_movielens_ratings(df_train, df_test):
    n_users = max(max(df_train.user_id.unique()), max(df_test.user_id.unique()))
    n_items = max(max(df_train.item_id.unique()), max(df_test.item_id.unique()))

    train_interactions = lil_matrix((n_users, n_items), dtype=float)
    test_interactions = lil_matrix((n_users, n_items), dtype=float)
    
    for train_row, test_row in zip(df_train.itertuples(), df_test.itertuples()):
        train_interactions[train_row[1] - 1, train_row[2] - 1] = train_row[3]
        test_interactions[test_row[1] - 1, test_row[2] - 1] = test_row[3]

    return train_interactions, test_interactions


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=3):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=False)
        # Also should consider fitting overall bias (self.mu term) and both user and item bias vectors
        # Mu is 1x1, user_bias is 1xn_users. item_bias is 1xn_items
    
    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
        pred = torch.mm(self.user_factors(users), torch.transpose(self.item_factors(items),0,1))
        return pred
    
    def get_variable(self, expression):
        var = Variable(expression)
        if torch.cuda.is_available():
            var = var.cuda()
        return var


class BiasedMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=3):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=False)
        self.user_biases = torch.nn.Embedding(n_users, 
                                              1,
                                              sparse=False)
        self.item_biases = torch.nn.Embedding(n_items,
                                              1,
                                              sparse=False)
        
    def forward(self, users, items):
        um = torch.mm(self.user_factors(users), torch.transpose(self.item_factors(items), 0, 1))
        um += self.item_biases(items).squeeze()
        pred = torch.transpose(torch.transpose(um, 0, 1) + self.user_biases(users).squeeze(), 0, 1)
        return pred
    
    def get_variable(self, expression):
        var = Variable(expression)
        if torch.cuda.is_available():
            var = var.cuda()
        return var


def get_batch(batch_size, ratings):
    # Sort our data and scramble it
    rows, cols = ratings.shape
    p = np.random.permutation(rows)
    
    # create batches
    sindex = 0
    eindex = batch_size
    while eindex < rows:
        batch = p[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= rows:
        batch = p[sindex:rows]
        yield batch


def run_epoch(model, ratings, reg_loss_func, test_mode=False):
    loss_func = torch.nn.MSELoss()
    BATCH_SIZE = 1000

    for i,batch in enumerate(get_batch(BATCH_SIZE, ratings)):
        # Set gradients to zero
        reg_loss_func.zero_grad()
        
        # Turn data into variables
        interactions = model.get_variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = model.get_variable(torch.LongTensor(batch))
        cols = model.get_variable(torch.LongTensor(np.arange(ratings.shape[1])))
    
        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
            
        if not test_mode:
            # Backpropagate
            loss.backward()

            # Update the parameters
            reg_loss_func.step()
        
    return loss.data[0]


def nonbiasedCV():
    EPOCH = 50
    lams = [0.1, 0.01, 0.001]

    losses = {}
    for lam in lams:
        print("L2 Regularization value - {}\n".format(lam))
        losses[lam] = []
        for r in range(1, 5):
            print("Downloading 'r{}' data...".format(r))
            train, test = get_datasets(r)
            print("Data downloaded")
            print("Transforming the data...")
            train, test = get_movielens_ratings(train, test)
            print("Data transformed")

            model = MatrixFactorization(train.shape[0],
                                        train.shape[1],
                                        n_factors=3)
            if torch.cuda.is_available():
                model = model.cuda()

            reg_loss_func = torch.optim.Adagrad(model.parameters(),
                                                weight_decay=lam)
            for i in range(EPOCH):
                tr_loss = np.sqrt(run_epoch(model, train, reg_loss_func))

                ts_loss = np.sqrt(run_epoch(model, test, reg_loss_func, test_mode=True))

                if (i + 1) % 5 == 0:
                    print("Epoch {}".format(i + 1))
                    print("Training Loss: {}".format(tr_loss))
                    print("Testing Loss: {}".format(ts_loss))

                losses[lam].append([tr_loss, ts_loss])
        print("Cross-validation RMSE is {} for reg {}".format(np.mean(np.asarray(losses[lam]),
                                                                      axis=0)[1],
                                                              lam))
        print("Standard Error: {}\n".format(stats.sem(np.asarray(losses[lam])[:, 1])))


def biasedCV():
    EPOCH = 50
    lams = [0.1, 0.01, 0.001]

    losses = {}
    for lam in lams:
        print("L2 Regularization value - {}\n".format(lam))
        losses[lam] = []
        for r in ['a', 'b']:
            print("Downloading 'r{}' data...".format(r))
            train, test = get_datasets(r)
            print("Data downloaded")
            print("Transforming the data...")
            train, test = get_movielens_ratings(train, test)
            print("Data transformed")

            model = BiasedMatrixFactorization(train.shape[0],
                                              train.shape[1],
                                              n_factors=3)
            if torch.cuda.is_available():
                model = model.cuda()

            reg_loss_func = torch.optim.Adagrad(model.parameters(),
                                                weight_decay=lam)
            print("Training...")
            for i in range(EPOCH):
                tr_loss = np.sqrt(run_epoch(model, train, reg_loss_func))

                ts_loss = np.sqrt(run_epoch(model, test, reg_loss_func, test_mode=True))

                if (i + 1) % 1 == 0:
                    print("Epoch {}".format(i + 1))
                    print("Training Loss: {}".format(tr_loss))
                    print("Testing Loss: {}".format(ts_loss))

            losses[lam].append([tr_loss, ts_loss])
        print("Cross-validation RMSE is {} for reg {}".format(np.mean(np.asarray(losses[lam]),
                                                                      axis=0)[1],
                                                              lam))
        print("Standard Error: {}\n".format(stats.sem(np.asarray(losses[lam])[:, 1])))


def train_final_model():
    train, test = get_datasets(5)
    print("Data downloaded")
    train, test = get_movielens_ratings(train, test)
    print("Data transformed")

    EPOCH = 100

    model = MatrixFactorization(train.shape[0],
                                train.shape[1],
                                n_factors=3)

    if torch.cuda.is_available():
        model = model.cuda()

    reg_loss_func = torch.optim.Adagrad(model.parameters(),
                                        weight_decay=0.1)
    for i in range(EPOCH):
        tr_loss = np.sqrt(run_epoch(model, train, reg_loss_func))

        ts_loss = np.sqrt(run_epoch(model, test, reg_loss_func, test_mode=True))

        if (i + 1) % 5 == 0:
            print("Epoch {}".format(i + 1))
            print("Training Loss: {}".format(tr_loss))
            print("Testing Loss: {}".format(ts_loss))
            torch.save(model, 'RecEngine.pt')


def make_predictions():
    model = torch.load('RecEngine.pt', map_location=lambda storage, loc: storage)

    raw_train, raw_test = get_datasets(5)
    print("Data downloaded")

    train, test = get_movielens_ratings(raw_train, raw_test)
    print("Data transformed")

    # Users which appear only in test set
    test_users = raw_test['user_id'].unique()
    # Movies which appear only in training set
    movies = set(raw_train['item_id'])

    seen_movies = []
    for arr in test[57374:]:
        if len(list(arr.nonzero()[1])) > 0:
            seen_movies.append(list(arr.nonzero()[1]))

    ratings = test[57374:]
    ratings = ratings[ratings.getnnz(1) > 0]

    # Turn data into variables
    rows = model.get_variable(torch.LongTensor(test_users-1))
    cols = model.get_variable(torch.LongTensor(np.arange(ratings.shape[1])))

    # Predict and calculate loss
    predictions = model(rows, cols).data.cpu().numpy()
    recs = []
    k = 0
    for seen, pred in zip(seen_movies, predictions):
        user_id = test_users[k]
        seen = set(seen)
        top_num = len(seen) + 5
        top = pred.argsort()[-top_num:][::-1]
        rec = []
        for ind in top:
            if ind not in seen and ind in movies:
                rec.append(str(ind))
                if len(rec) >= 5:
                    recs.append([str(user_id)] + rec)
                    break
        k += 1
        if k % 1000 == 0:
            print(user_id)

    writer = tsv.TsvWriter(open("assign5_r5results.tsv", "w"))
    for rec in recs:
        writer.list_line(rec)
    writer.close()


if __name__ == "__main__":
    nonbiasedCV()
    biasedCV()
    train_final_model()
    make_predictions()
