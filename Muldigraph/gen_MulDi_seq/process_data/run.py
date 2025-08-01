import os
import shutil
import sys


path = "/home/hainguyen/TLmGNN/Muldigraph/gen_MulDi_seq"
os.chdir(path)

os.makedirs('./trans_sentence', exist_ok=True)
for i in range(1,7):
    file_path = f'./process_data/dataset{i}.py'
    if os.path.exists(file_path):
        try:
             os.system( f'python {file_path}')
        except:
            sys.exit()

os.makedirs('./trans_sentence', exist_ok=True)
