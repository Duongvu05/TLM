from tqdm import tqdm
import pickle
import os  
import functools
import pandas as pd
import random 

def read_pkl(pkl_file):
    print(f'Reading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(data, pkl_file):
    print(f'Saving data to {pkl_file}...')
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def load_and_print_pkl(pkl_file):
    print(f'Loading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    i = 0
    for account, trans in enumerate(data):
            if i == 3:
                break
            print(trans)

def save_txt(data,txt_file):
    with open(txt_file,"w",encoding="utf-8") as file:
        for account in data:
            file.write(f"{account}\n")

    return
        
def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0
    
def extract_transactions(G):
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'accounts_data_generate'):
        amount = tnx_info['amount']
        block_timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']
        transaction = {
            'tag': tag,
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': block_timestamp,
        }
        transactions.append(transaction)
    return transactions

def filtering(transactions):
    f_in = {}
    f_out = {}
    error_tran = []
    for tran in transactions: 
        tag = tran['tag']
        from_address = tran['from_address']
        to_address = tran['to_address']
        amount = tran['amount']
        block_timestamp = tran['timestamp']
        if from_address == "" or to_address == "":
            error_tran.append(tran)
            continue
        try:
            f_out[from_address].append([to_address, block_timestamp, amount, "OUT", tag, 1])
        except KeyError:
            f_out[from_address] = [[to_address, block_timestamp, amount, "OUT", tag, 1]]

        try:
            f_in[to_address].append([from_address, block_timestamp, amount, "IN", tag, 1])
        except KeyError:
            f_in[to_address] = [[from_address, block_timestamp, amount, "IN", tag, 1]]

    return f_in, f_out


def seq_generation(eoa2seq_in, eoa2seq_out):

    eoa_list = list(eoa2seq_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += 1
            # if cnt_all >= 5 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq

def create_phisher_account(processed_data):
    phisher_accounts = []
    normal_accounts = []
    for address, txs in tqdm(processed_data.items(),desc="Filtering Abnormal vs Normal accounts"):
        i = 0
        for tx in txs:
            if tx[4] == 1:
                phisher_accounts.append(address)
                i += 1
                break
            else: 
                continue
        
        if i == 0: 
            normal_accounts.append(address)
    
    return (phisher_accounts,normal_accounts)

def data_generate():
    graph_file = './raw_data/MulDiGraph.pkl'
    os.makedirs('./data_train_MulDi', exist_ok=True)
    out_file = './data_train_MulDi/eoa2seq.pkl'

    graph = read_pkl(graph_file)
    transactions = extract_transactions(graph)
    eoa2seq_in, eoa2seq_out = filtering(transactions)
    eoa2seq_agg = seq_generation(eoa2seq_in,eoa2seq_out)
    phisher_accounts , normal_accounts = create_phisher_account(eoa2seq_agg)
    print(f"Number of phisher account: {len(phisher_accounts)}")
    save_txt(set(phisher_accounts),"./raw_data/phisher_accounts.txt")
    
    """
    Ratio 5:5
    """
    selected_account = random.sample(normal_accounts, len(phisher_accounts))
    final_account = selected_account + phisher_accounts
    eoa2seq_final = {account:eoa2seq_agg[account] for account in final_account}
    save_pkl(eoa2seq_final, out_file)

if __name__ == '__main__':
    data_generate()
    pkl_file = 'data_train_MulDi/transactions1.pkl'
    load_and_print_pkl(pkl_file)

