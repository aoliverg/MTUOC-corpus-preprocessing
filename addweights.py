import codecs
import sys


fentrada=sys.argv[1]
fsortida=sys.argv[2]
weight=sys.argv[3]

entrada=codecs.open(fentrada,"r",encoding="utf-8")
sortida=codecs.open(fsortida,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.strip()
    linia=linia+"\t"+str(weight)
    sortida.write(linia+"\n")
entrada.close()
sortida.close()
