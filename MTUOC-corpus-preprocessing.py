#    MTUOC-corpus-preprocessing
#    Copyright (C) 2022  Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
from datetime import datetime
import os
import codecs
import importlib
import re

import pickle

from shutil import copyfile

import yaml
from yaml import load, dump


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def file_len(fname):
    num_lines = sum(1 for line in open(fname))
    return(num_lines)
    
def findEMAILs(string): 
    email=re.findall('\S+@\S+', string)   
    return email
    
def findURLs(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
    
def replace_EMAILs(string,code="@EMAIL@"):
    EMAILs=findEMAILs(string)
    cont=0
    for EMAIL in EMAILs:
        string=string.replace(EMAIL,code)
    return(string)

def replace_URLs(string,code="@URL@"):
    URLs=findURLs(string)
    cont=0
    for URL in URLs:
        string=string.replace(URL,code)
    return(string)

stream = open('config-corpus-preprocessing.yaml', 'r',encoding="utf-8")
config=yaml.load(stream, Loader=yaml.FullLoader)

MTUOC=config["MTUOC"]
sys.path.append(MTUOC)

from MTUOC_train_truecaser import TC_Trainer
from MTUOC_truecaser import Truecaser
from MTUOC_splitnumbers import splitnumbers

import sentencepiece as spm

from MTUOC_sentencepiece import sentencepiece_train
from MTUOC_sentencepiece import sentencepiece_encode
from MTUOC_subwordnmt import subwordnmt_train
from MTUOC_subwordnmt import subwordnmt_encode

preprocess_type=config["preprocess_type"]

corpus=config["corpus"]
valsize=int(config["valsize"])
evalsize=int(config["evalsize"])
train_weightsFile=config["train_weights"]
val_weightsFile=config["val_weights"]
SLcode3=config["SLcode3"]
SLcode2=config["SLcode2"]
TLcode3=config["TLcode3"]
TLcode2=config["TLcode2"]

from_train_val=config["from_train_val"]
train_corpus=config["train_corpus"]
val_corpus=config["val_corpus"]

#VERBOSE
VERBOSE=config["VERBOSE"]
LOGFILE=config["LOG_FILE"]

REPLACE_EMAILS=config["REPLACE_EMAILS"]
EMAIL_CODE=config["EMAIL_CODE"]
REPLACE_URLS=config["REPLACE_URLS"]
URL_CODE=config["URL_CODE"]


TRAIN_SL_TRUECASER=config["TRAIN_SL_TRUECASER"]
SL_DICT=config["SL_DICT"]
TRUECASE_SL=config["TRUECASE_SL"]
SL_TC_MODEL=config["SL_TC_MODEL"]
if SL_TC_MODEL=="auto":
    SL_TC_MODEL="tc."+SLcode2

TRAIN_TL_TRUECASER=config["TRAIN_TL_TRUECASER"]
TL_DICT=config["TL_DICT"]
TRUECASE_TL=config["TRUECASE_TL"]
TL_TC_MODEL=config["TL_TC_MODEL"]
if TL_TC_MODEL=="auto":
    TL_TC_MODEL="tc."+TLcode2
    
SL_TOKENIZER=config["SL_TOKENIZER"]
if SL_TOKENIZER=="None":
    SL_TOKENIZER=None
TL_TOKENIZER=config["TL_TOKENIZER"]
if TL_TOKENIZER=="None":
    TL_TOKENIZER=None
TOKENIZE_SL=config["TOKENIZE_SL"]
TOKENIZE_TL=config["TOKENIZE_TL"]

if SL_TOKENIZER==None: TOKENIZE_SL=False
if TL_TOKENIZER==None: TOKENIZE_TL=False


CLEAN=config["CLEAN"]
MIN_TOK=config["MIN_TOK"]
MAX_TOK=config["MAX_TOK"]

MIN_CHAR=config["MIN_CHAR"]
MAX_CHAR=config["MAX_CHAR"]

#SENTENCE PIECE
SP_MODEL_PREFIX=config["SP_MODEL_PREFIX"]
MODEL_TYPE=config["MODEL_TYPE"]
#one of unigram, bpe, char, word
JOIN_LANGUAGES=config["JOIN_LANGUAGES"]
VOCAB_SIZE=config["VOCAB_SIZE"]
CHARACTER_COVERAGE=config["CHARACTER_COVERAGE"]
CHARACTER_COVERAGE_SL=config["CHARACTER_COVERAGE_SL"]
CHARACTER_COVERAGE_TL=config["CHARACTER_COVERAGE_TL"]
VOCABULARY_THRESHOLD=config["VOCABULARY_THRESHOLD"]
INPUT_SENTENCE_SIZE=config["INPUT_SENTENCE_SIZE"]
CONTROL_SYMBOLS=config["CONTROL_SYMBOLS"]
USER_DEFINED_SYMBOLS=config["USER_DEFINED_SYMBOLS"]

BOS=config["bos"]
EOS=config["eos"]

#SUBWORD NMT
LEARN_BPE=config["LEARN_BPE"]
JOINER=config["JOINER"]
SPLIT_DIGITS=config["SPLIT_DIGITS"]
NUM_OPERATIONS=config["NUM_OPERATIONS"]
APPLY_BPE=config["APPLY_BPE"]
BPE_DROPOUT=config["BPE_DROPOUT"]
BPE_DROPOUT_P=config["BPE_DROPOUT_P"]

#GUIDED ALIGNMENT
#TRAIN CORPUS
GUIDED_ALIGNMENT=config["GUIDED_ALIGNMENT"]
ALIGNER=config["ALIGNER"]
#one of eflomal, fast_align

DELETE_EXISTING=config["DELETE_EXISTING"]
DELETE_TEMP=config["DELETE_TEMP"]
SPLIT_LIMIT=config["SPLIT_LIMIT"]
#For efomal, max number of segments to align at a time

#VALID CORPUS
GUIDED_ALIGNMENT_VALID=config["GUIDED_ALIGNMENT_VALID"]
ALIGNER_VALID=config["ALIGNER_VALID"]
#one of eflomal, fast_align
#BOTH_DIRECTIONS_VALID: True 
#only for fast_align, eflomal aligns always the two directions at the same time
DELETE_EXISTING_VALID=config["DELETE_EXISTING_VALID"]

#simalign
simalign_device=config["simalign"]["device"]
simalign_matching_method=config["simalign"]["matching_method"]


if VERBOSE:
    logfile=codecs.open(LOGFILE,"w",encoding="utf-8")

if not from_train_val:
    #SPLITTING CORPUS
    from MTUOC_train_val_eval import split_corpus
    if VERBOSE:
        cadena="Splitting corpus: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    split_corpus(corpus,valsize,evalsize,SLcode3,TLcode3)

    trainCorpus="train-"+SLcode3+"-"+TLcode3+".txt"
    valCorpus="val-"+SLcode3+"-"+TLcode3+".txt"
    evalCorpus="eval-"+SLcode3+"-"+TLcode3+".txt"
    trainPreCorpus="train-pre-"+SLcode3+"-"+TLcode3+".txt"
    valPreCorpus="val-pre-"+SLcode3+"-"+TLcode3+".txt"
    evalSL="eval."+SLcode2
    evalTL="eval."+TLcode2
    entrada=codecs.open(evalCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open(evalSL,"w",encoding="utf-8")
    sortidaTL=codecs.open(evalTL,"w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
    
else:
    trainCorpus=train_corpus
    valCorpus=val_corpus
    trainPreCorpus="train-pre-"+SLcode3+"-"+TLcode3+".txt"
    valPreCorpus="val-pre-"+SLcode3+"-"+TLcode3+".txt"


if VERBOSE:
    cadena="Start of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

#TRAIN

entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open("trainSL.temp","w",encoding="utf-8")
sortidaTL=codecs.open("trainTL.temp","w",encoding="utf-8")
sortidaW=codecs.open(train_weightsFile,"w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
        if len(camps)>=3:
            sortidaW.write(camps[2]+"\n")
        else:
            sortidaW.write("\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()

if TRAIN_SL_TRUECASER:
    if VERBOSE:
        cadena="Training SL Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    SLTrainer=TC_Trainer(MTUOC, SL_TC_MODEL, "trainSL.temp", SL_DICT, SL_TOKENIZER)
    SLTrainer.train_truecaser()

if TRAIN_TL_TRUECASER:
    if VERBOSE:
        cadena="Training TL Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    TLTrainer=TC_Trainer(MTUOC, TL_TC_MODEL, "trainTL.temp", TL_DICT, TL_TOKENIZER)
    TLTrainer.train_truecaser()    

if TRUECASE_SL:
    truecaserSL=Truecaser()
    truecaserSL.set_MTUOCPath(MTUOC)
    truecaserSL.set_tokenizer(SL_TOKENIZER)
    truecaserSL.set_tc_model(SL_TC_MODEL)

if TRUECASE_TL:
    truecaserTL=Truecaser()
    truecaserTL.set_MTUOCPath(MTUOC)
    truecaserTL.set_tokenizer(TL_TOKENIZER)
    truecaserTL.set_tc_model(TL_TC_MODEL)


if not SL_TOKENIZER==None:
    SL_TOKENIZER=MTUOC+"/"+SL_TOKENIZER
    if not SL_TOKENIZER.endswith(".py"): SL_TOKENIZER=SL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', SL_TOKENIZER)
    tokenizerSLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerSLmod)
    tokenizerSL=tokenizerSLmod.Tokenizer()

if not TL_TOKENIZER==None: 
    TL_TOKENIZER=MTUOC+"/"+TL_TOKENIZER   
    if not TL_TOKENIZER.endswith(".py"): TL_TOKENIZER=TL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', TL_TOKENIZER)
    tokenizerTLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerTLmod)
    tokenizerTL=tokenizerTLmod.Tokenizer()



if VERBOSE:
    cadena="Preprocessing train corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")


entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortida=codecs.open(trainPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_SL:
            toksl=tokenizerSL.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_TL:
            toktl=tokenizerTL.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_SL:
                toksl=truecaserSL.truecase(toksl)
            if TRUECASE_TL:
                toktl=truecaserTL.truecase(toktl)
            if not weight==None:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            else:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()
#Val CORPUS
if VERBOSE:
    cadena="Preprocessing val corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

entrada=codecs.open(valCorpus,"r",encoding="utf-8")
sortida=codecs.open(valPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_SL:
            toksl=tokenizerSL.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_TL:
            toktl=tokenizerTL.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_SL:
                toksl=truecaserSL.truecase(toksl)
            if TRUECASE_TL:
                toktl=truecaserTL.truecase(toktl)
            
            if not weight==None:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            else:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()



if preprocess_type=="sentencepiece":
    ###sentencepiece is default if no smt or subword-nmt is selected
    if VERBOSE:
        cadena="Start of sentencepiece process: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")

    if VERBOSE:
        cadena="Start of sentencepiece training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")

    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("trainPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("trainPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")

        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
    sortidaW.close()
            
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("valPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("valPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
        
    if VERBOSE:
        cadena="Training sentencepiece: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    bosSP=True
    eosSP=True
    if BOS=="None": bosSP=False
    if EOS=="None": eosSP=False
    sentencepiece_train("trainPreSL.temp","trainPreTL.temp",SLcode2=SLcode2,TLcode2=TLcode2,JOIN_LANGUAGES=JOIN_LANGUAGES,SP_MODEL_PREFIX=SP_MODEL_PREFIX,MODEL_TYPE=MODEL_TYPE,VOCAB_SIZE=VOCAB_SIZE,CHARACTER_COVERAGE=CHARACTER_COVERAGE,INPUT_SENTENCE_SIZE=INPUT_SENTENCE_SIZE,SPLIT_DIGITS=SPLIT_DIGITS,CONTROL_SYMBOLS=CONTROL_SYMBOLS,USER_DEFINED_SYMBOLS=USER_DEFINED_SYMBOLS)
    
    if VERBOSE:
        cadena="Encoding corpora with sentencepiece: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    
    if JOIN_LANGUAGES:
        SP_MODEL=SP_MODEL_PREFIX+".model"
    else:
        SP_MODEL=SP_MODEL_PREFIX+"-"+SLcode2+".model"
    
    outfile="train.sp."+SLcode2
    vocabulary_file="vocab_file."+SLcode2
    sentencepiece_encode("trainPreSL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    outfile="val.sp."+SLcode2
    vocabulary_file="vocab_file."+SLcode2
    sentencepiece_encode("valPreSL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    
    if JOIN_LANGUAGES:
        SP_MODEL=SP_MODEL_PREFIX+".model"
    else:
        SP_MODEL=SP_MODEL_PREFIX+"-"+TLcode2+".model"
    
    outfile="train.sp."+TLcode2
    vocabulary_file="vocab_file."+TLcode2
    sentencepiece_encode("trainPreTL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    outfile="val.sp."+TLcode2
    sentencepiece_encode("valPreTL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)

    if GUIDED_ALIGNMENT:
        if VERBOSE:
            cadena="Guided alignment training: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="train.sp."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
        if ALIGNER=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"train.sp","train.sp","trainPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_eflomal(MTUOC,"train.sp","train.sp","trainPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("train.sp."+SLcode2,"train.sp."+TLcode2,"train.sp."+SLcode2+"."+TLcode2+".align")
        
        elif ALIGNER=="awesome":
            sys.path.append(MTUOC)
            from MTUOC_awesome_aligner import *
            awesome_finetune=config["awesome"]["finetune"]
            awesome_finetune_limit=config["awesome"]["finetune_limit"]
            if awesome_finetune:
                FILET1="train.sp."+SLcode2
                FILET2="train.sp."+TLcode2
                FILEV1="val.sp."+SLcode2
                FILEV2="val.sp."+TLcode2
                if not awesome_finetune_limit==-1:
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET1+" > awesometrain."+SLcode2+".temp"
                    os.system(command)
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET2+" > awesometrain."+TLcode2+".temp"
                    os.system(command)
                    FILET1="awesometrain."+SLcode2+".temp"
                    FILET2="awesometrain."+TLcode2+".temp"
                    
                    
                awesome_finetune_initial_model=config["awesome"]["finetune_initial_model"] 
                awesome_finetuned_dir=config["awesome"]["finetuned_dir"]
                awesome_finetune_device=config["awesome"]["finetune_device"]
                awesome_finetune_cuda_visible_devices=str(config["awesome"]["finetune_cuda_visible_devices"])
                finetune_awesome(FILET1, FILET2, FILEV1, FILEV2, initial_model=awesome_finetune_initial_model, output_dir=awesome_finetuned_dir,device=awesome_finetune_device, cuda_visible_devices=awesome_finetune_cuda_visible_devices)
                os.remove("awesometrain."+SLcode2+".temp")
                os.remove("awesometrain."+TLcode2+".temp")
            FILE1="train.sp."+SLcode2
            FILE2="train.sp."+TLcode2
            OUTPUT_FILE="train.sp."+SLcode2+"."+TLcode2+".align"
            awesome_align_model=config["awesome"]["align_model"]
            awesome_align_device=config["awesome"]["align_device"]
            awesome_align_cuda_visible_devices=str(config["awesome"]["align_cuda_visible_devices"])
            align_awesome_aligner(FILE1, FILE2, OUTPUT_FILE, model=awesome_align_model, device=awesome_align_device, cuda_visible_devices=awesome_align_cuda_visible_devices)
        
        

    if GUIDED_ALIGNMENT_VALID:
        if VERBOSE:
                cadena="Guided alignment valid: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="val.sp."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
            FILE="val.sp."+TLcode2+"."+TLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)            
        if ALIGNER_VALID=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            
        elif ALIGNER_VALID=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            guided_alignment_eflomal(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("val.sp."+SLcode2,"val.sp."+TLcode2,"val.sp."+SLcode2+"."+TLcode2+".align")
                
elif preprocess_type=="subwordnmt":
    print("SUBWORD NMT BPE")
    #####################
    print("Starting BPE training",datetime.now())

    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("trainPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("trainPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
            
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("valPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("valPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()

    if LEARN_BPE: 
        if VERBOSE:
            print("Learning BPE",datetime.now())
        if JOIN_LANGUAGES: 
            if VERBOSE: print("JOINING LANGUAGES",datetime.now())
            subwordnmt_train("trainPreSL.temp trainPreTL.temp",SLcode2=SLcode2,TLcode2=TLcode2,NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file")

        else:
            print("**************NOT JOINING LANGUAGES")
            if VERBOSE: print("SL",datetime.now())
            subwordnmt_train("trainPreSL.temp",SLcode2=SLcode2,TLcode2="",NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file."+SLcode2)
           
            if VERBOSE: print("TL",datetime.now())
            subwordnmt_train("trainPreTL.temp",SLcode2=TLcode2,TLcode2="",NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file."+TLcode2)
           


    if APPLY_BPE: 
        if VERBOSE:
            print("Applying BPE",datetime.now())
        if JOIN_LANGUAGES:
            BPESL="codes_file"
            BPETL="codes_file"
        if not JOIN_LANGUAGES:
            BPESL="codes_file."+SLcode2
            BPETL="codes_file."+TLcode2
        
        subwordnmt_encode("trainPreSL.temp","train.bpe."+SLcode2,CODES_FILE=BPESL,VOCAB_FILE="vocab_BPE."+SLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        subwordnmt_encode("trainPreTL.temp","train.bpe."+TLcode2,CODES_FILE=BPETL,VOCAB_FILE="vocab_BPE."+TLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        
        subwordnmt_encode("valPreSL.temp","val.bpe."+SLcode2,CODES_FILE=BPESL,VOCAB_FILE="vocab_BPE."+SLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        subwordnmt_encode("valPreTL.temp","val.bpe."+TLcode2,CODES_FILE=BPETL,VOCAB_FILE="vocab_BPE."+TLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
       
            
   
    
    if GUIDED_ALIGNMENT:
        if VERBOSE:
            cadena="Guided alignment training: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="train.bpe."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
        if ALIGNER=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"train.bpe","train.bpe","trainPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_eflomal(MTUOC,"train.bpe","train.bpe","trainPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("train.bpe."+SLcode2,"train.bpe."+TLcode2,"train.bpe."+SLcode2+"."+TLcode2+".align")
        
        elif ALIGNER=="awesome":
            sys.path.append(MTUOC)
            from MTUOC_awesome_aligner import *
            awesome_finetune=config["awesome"]["finetune"]
            awesome_finetune_limit=config["awesome"]["finetune_limit"]
            if awesome_finetune:
                FILET1="train.bpe."+SLcode2
                FILET2="train.bpe."+TLcode2
                FILEV1="val.bpe."+SLcode2
                FILEV2="val.bpe."+TLcode2
                if not awesome_finetune_limit==-1:
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET1+" > awesometrain."+SLcode2+".temp"
                    os.system(command)
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET2+" > awesometrain."+TLcode2+".temp"
                    os.system(command)
                    FILET1="awesometrain."+SLcode2+".temp"
                    FILET2="awesometrain."+TLcode2+".temp"
                    
                    
                awesome_finetune_initial_model=config["awesome"]["finetune_initial_model"] 
                awesome_finetuned_dir=config["awesome"]["finetuned_dir"]
                awesome_finetune_device=config["awesome"]["finetune_device"]
                awesome_finetune_cuda_visible_devices=str(config["awesome"]["finetune_cuda_visible_devices"])
                finetune_awesome(FILET1, FILET2, FILEV1, FILEV2, initial_model=awesome_finetune_initial_model, output_dir=awesome_finetuned_dir,device=awesome_finetune_device, cuda_visible_devices=awesome_finetune_cuda_visible_devices)
                os.remove("awesometrain."+SLcode2+".temp")
                os.remove("awesometrain."+TLcode2+".temp")
            FILE1="train.bpe."+SLcode2
            FILE2="train.bpe."+TLcode2
            OUTPUT_FILE="train.bpe."+SLcode2+"."+TLcode2+".align"
            awesome_align_model=config["awesome"]["align_model"]
            awesome_align_device=config["awesome"]["align_device"]
            awesome_align_cuda_visible_devices=str(config["awesome"]["align_cuda_visible_devices"])
            align_awesome_aligner(FILE1, FILE2, OUTPUT_FILE, model=awesome_align_model, device=awesome_align_device, cuda_visible_devices=awesome_align_cuda_visible_devices)
        
        

    if GUIDED_ALIGNMENT_VALID:
        if VERBOSE:
                cadena="Guided alignment valid: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="val.bpe."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
            FILE="val.bpe."+TLcode2+"."+TLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)            
        if ALIGNER_VALID=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            
        elif ALIGNER_VALID=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            guided_alignment_eflomal(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("val.bpe."+SLcode2,"val.bpe."+TLcode2,"val.bpe."+SLcode2+"."+TLcode2+".align")
                
        
    
    
    #####################


elif preprocess_type=="smt":
    #train
    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    nomsl="train.smt."+SLcode2
    nomtl="train.smt."+TLcode2
    sortidaSL=codecs.open(nomsl,"w",encoding="utf-8")
    sortidaTL=codecs.open(nomtl,"w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        try:
            camps=linia.split("\t")
            SLsegment=camps[0]
            TLsegment=camps[1]
            sortidaSL.write(SLsegment+"\n")
            sortidaTL.write(TLsegment+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        except:
            pass
            
    #val
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    nomsl="val.smt."+SLcode2
    nomtl="val.smt."+TLcode2
    sortidaSL=codecs.open(nomsl,"w",encoding="utf-8")
    sortidaTL=codecs.open(nomtl,"w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        try:
            camps=linia.split("\t")
            SLsegment=camps[0]
            TLsegment=camps[1]
            sortidaSL.write(SLsegment+"\n")
            sortidaTL.write(TLsegment+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        except:
            pass
            
    if GUIDED_ALIGNMENT:
        if VERBOSE:
            cadena="Guided alignment training: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="train.smt."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
        if ALIGNER=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"train.smt","train.smt","trainPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_eflomal(MTUOC,"train.smt","train.smt","trainPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("trainPreW.temp",train_weightsFile)
            
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("train.smt."+SLcode2,"train.smt."+TLcode2,"train.smt."+SLcode2+"."+TLcode2+".align")
        
        elif ALIGNER=="awesome":
            sys.path.append(MTUOC)
            from MTUOC_awesome_aligner import *
            awesome_finetune=config["awesome"]["finetune"]
            awesome_finetune_limit=config["awesome"]["finetune_limit"]
            if awesome_finetune:
                FILET1="train.smt."+SLcode2
                FILET2="train.smt."+TLcode2
                FILEV1="val.smt."+SLcode2
                FILEV2="val.smt."+TLcode2
                if not awesome_finetune_limit==-1:
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET1+" > awesometrain."+SLcode2+".temp"
                    os.system(command)
                    command="head -n "+str(awesome_finetune_limit)+" "+FILET2+" > awesometrain."+TLcode2+".temp"
                    os.system(command)
                    FILET1="awesometrain."+SLcode2+".temp"
                    FILET2="awesometrain."+TLcode2+".temp"
                    
                    
                awesome_finetune_initial_model=config["awesome"]["finetune_initial_model"] 
                awesome_finetuned_dir=config["awesome"]["finetuned_dir"]
                awesome_finetune_device=config["awesome"]["finetune_device"]
                awesome_finetune_cuda_visible_devices=str(config["awesome"]["finetune_cuda_visible_devices"])
                finetune_awesome(FILET1, FILET2, FILEV1, FILEV2, initial_model=awesome_finetune_initial_model, output_dir=awesome_finetuned_dir,device=awesome_finetune_device, cuda_visible_devices=awesome_finetune_cuda_visible_devices)
                os.remove("awesometrain."+SLcode2+".temp")
                os.remove("awesometrain."+TLcode2+".temp")
            FILE1="train.smt."+SLcode2
            FILE2="train.smt."+TLcode2
            OUTPUT_FILE="train.smt."+SLcode2+"."+TLcode2+".align"
            awesome_align_model=config["awesome"]["align_model"]
            awesome_align_device=config["awesome"]["align_device"]
            awesome_align_cuda_visible_devices=str(config["awesome"]["align_cuda_visible_devices"])
            align_awesome_aligner(FILE1, FILE2, OUTPUT_FILE, model=awesome_align_model, device=awesome_align_device, cuda_visible_devices=awesome_align_cuda_visible_devices)
        
        

    if GUIDED_ALIGNMENT_VALID:
        if VERBOSE:
                cadena="Guided alignment valid: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        if DELETE_EXISTING:
            FILE="val.smt."+SLcode2+"."+SLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)
            FILE="val.smt."+TLcode2+"."+TLcode2+".align" 
            if os.path.exists(FILE):
                os.remove(FILE)            
        if ALIGNER_VALID=="fast_align":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
            if VERBOSE:
                cadena="Fast_align: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
            guided_alignment_fast_align(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,False,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            
        elif ALIGNER_VALID=="eflomal":
            sys.path.append(MTUOC)
            from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
            guided_alignment_eflomal(MTUOC,"val.sp","val.sp","valPreW.temp",SLcode2,TLcode2,SPLIT_LIMIT,VERBOSE)
            copyfile("valPreW.temp",val_weightsFile)
            if VERBOSE:
                cadena="Eflomal: "+str(datetime.now())
                print(cadena)
                logfile.write(cadena+"\n")
        
        elif ALIGNER=="simalign":
            sys.path.append(MTUOC)
            from MTUOC_simalign import *
            aligner=MTUOC_simalign(device=simalign_device, matching_method=simalign_matching_method,sltokenizer=None,tltokenizer=None)
            aligner.align_files("val.smt."+SLcode2,"val.smt."+TLcode2,"val.smt."+SLcode2+"."+TLcode2+".align")
            

if VERBOSE:
    cadena="End of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

#DELETE TEMPORAL FILES


if DELETE_TEMP:
    if VERBOSE:
        cadena="Deleting temporal files: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    todeletetemp=["trainSL.temp","trainTL.temp","trainPreSL.temp","trainPreTL.temp","valPreSL.temp","valPreTL.temp","train-pre-"+SLcode3+"-"+TLcode3+".txt","val-pre-"+SLcode3+"-"+TLcode3+".txt"]
    for td in todeletetemp:
        try:
            os.remove(td)
        except:
            pass

