from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import ByteLevel
import os
from tqdm import tqdm
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "True"

file_paths =[]
base_path = 'D:/Data/'
dirs = os.listdir(base_path)
i = 0
for dir in tqdm(dirs,total=len(dirs)):
    files = os.listdir(os.path.join(base_path,dir))
    for f in files:
        if f.find('.txt') > -1 or f.find('.md') > -1:
            f_path = base_path+dir+'/'+f
            file_paths.append(f_path)
            
unk_token = "[UNK]"  # token for unknown words
spl_tokens = ["\n","\r","\t","{","}","[","]","<",">","(",")","%","$","&","+","/","\\","|","_","*",";","=","-","[UNK]","[PAD]","[SEP]","[MASK]","[CLS]","[END_GEN]"]  # special tokens

def prepare_tokenizer_trainer():

    tokenizer = Tokenizer(BPE(unk_token = unk_token))
    trainer = BpeTrainer(vocab_size = 110000, show_progress=True, special_tokens = spl_tokens)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer, trainer

def file_iterator():
    for index in tqdm(range(0,len(file_paths),20),total=int(len(file_paths)/20)):
        gc.collect()
        if index + 20 < len(file_paths) -1:
            paths = file_paths[index:index+10]
        else:
            paths = file_paths[index:]
        texts = []
        for path in paths:
            with open(path,"r",errors='ignore') as f:
                out = f.read()
            texts.append(out)
        yield texts
        
def train_tokenizer():
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer()
    tokenizer.train_from_iterator(file_iterator(), trainer=trainer) # training the tokenzier
    tokenizer.save("./tokenizer-trained.json")
    tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
    return tokenizer

train_tokenizer()        
