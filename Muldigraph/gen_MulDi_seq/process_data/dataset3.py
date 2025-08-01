import pickle as pkl
from tqdm import tqdm 

with open("data_train_MulDi/transactions1.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in tqdm(trans_seq.items(), desc="Sorting transaction lists"):
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

with open("data_train_MulDi/transactions2.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)
