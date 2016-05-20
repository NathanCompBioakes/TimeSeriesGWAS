#!/usr/bin/env python
import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import log10
plt.style.use('ggplot')

def main(argv):
    df = pd.read_table(argv[1], header=None)
    df = df[df > 0]
    #ggplot(aes(x=range(len(df)), y=-1*df[[3]].applymap(log10), color=df[[1]]), data=df) + geom_point()
    plt.scatter(range(len(df)), -1*df[[3]].applymap(log10), c=df[[1]])
    plt.axhline(y=-log10(0.00000005), xmin=0, xmax=len(df), hold=None, color='r')
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(p)')
    plt.savefig('plot.png')
    #plt.show()



if __name__=="__main__":
    main(sys.argv)
