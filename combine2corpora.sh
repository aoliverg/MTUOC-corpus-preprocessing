#!/bin/bash      

L1=eng
L2=spa
CORPUS1=ECB-eng-spa.txt
CORPUS2=GlobalVoices-eng-spa.txt
WEIGHT1=1
WEIGHT2=0.5
VALSIZE=5000
EVALSIZE=5000

sed s/$/\\t$WEIGHT1/ $CORPUS1 > corpusA.temp
sed s/$/\\t$WEIGHT2/ $CORPUS2 > corpusB.temp

LINES=$(wc -l corpusA.temp | cut -d " " -f 1)

TRAINSIZE=$((LINES-VALSIZE-EVALSIZE))

head -n $TRAINSIZE corpusA.temp > train.temp
tail -n $((VALSIZE+EVALSIZE)) corpusA.temp | head -n $VALSIZE > val-$L1-$L2.txt
tail -n $EVALSIZE corpusA.temp > eval-$L1-$L2.txt

cat train.temp corpusB.temp | sort | uniq | shuf > train-$L1-$L2.txt

cut -f 3 train-$L1-$L2.txt > train-weights.txt
cut -f 3 val-$L1-$L2.txt > val-weights.txt

cp train-$L1-$L2.txt A.temp
cp val-$L1-$L2.txt B.temp

cut -f 1,2 A.temp > train-$L1-$L2.txt
cut -f 1,2 B.temp > val-$L1-$L2.txt

rm *.temp