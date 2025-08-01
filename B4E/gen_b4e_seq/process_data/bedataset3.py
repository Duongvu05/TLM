import pickle as pkl
from tqdm import tqdm

with open("data_train_b4e/bert4eth_trans2.pkl", "rb") as f:
    trans_seq = pkl.load(f)

def mapping(eoa):
    io = eoa[3]
    if io == 'IN':
        eoa[3] = 0
    elif io == 'OUT':
        eoa[3] = 1

    return eoa
    
def feature_bucklization(eoa2seq):
    for address, transactions in tqdm(eoa2seq.items(),desc="Featrue bucklization"):
        new_transactions = list(map(lambda x: mapping(x), transactions))

        eoa2seq[address] = new_transactions
    return eoa2seq

trans_seq = feature_bucklization(trans_seq)

with open("data_train_b4e/bert4eth_trans3.pkl", "wb") as f:
    pkl.dump(trans_seq, f)
print(1)


