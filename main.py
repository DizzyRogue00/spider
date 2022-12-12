import pandas as pd
import numpy as np
import os
from functools import reduce

clips = [
    "BV1Jo4y127ob",
    "BV18W4y1b7py",
    "BV1w64y127Ui",
    "BV1FL4y1L7zM",
    "BV1TB4y1u785",
    "BV1zr4y1a7Sc",
    "BV1bK4y1K7yv",
    "BV1TP4y1x7Pp",
    "BV1Xf4y1j7Ps",
    "BV1So4y1f7gD",
    "BV1Av411j77t",
    "BV1Dh41187Hy"
]
files =os.listdir()

def ExtractFile(item,files):
    l=[s for s in files if item in s]
    for ll in l:
        if "Exact" in ll:
            l1=ll
        else:
            l2=ll
    return l1,l2

def fileList(clips,files):
    l1=[pd.read_csv(ExtractFile(item,files)[0],header=0) for item in clips]
    l2=[pd.read_csv(ExtractFile(item,files)[1],header=0)[['send_time','like','text']] for item in clips]
    return l1,l2

danmakuList,commentList=fileList(clips,files)
danmaku_data=pd.concat(danmakuList)
comment_data=pd.concat(commentList)
danmaku_data=danmaku_data.reset_index(drop=True)
comment_data=comment_data.reset_index(drop=True)
danmaku_data.to_csv('Bulletin-screen data.csv',index=False)
comment_data.to_csv('Comment data.csv',index=False)

danmaku_label=danmaku_data.sample(n=200,random_state=1)
comment_label=comment_data.sample(n=200,random_state=1)

danmaku_rest=danmaku_data[~danmaku_data.index.isin(danmaku_label.index)]
comment_rest=comment_data[~comment_data.index.isin(comment_label.index)]

danmaku_label1=danmaku_label.reset_index(drop=True)
danmaku_rest1=danmaku_rest.reset_index(drop=True)
comment_label1=comment_label.reset_index(drop=True)
comment_rest1=comment_rest.reset_index(drop=True)

danmaku_label1.to_csv("Bulletin-screen 200 label data.csv",index=False)
danmaku_rest1.to_csv("Bulletin-screen 200 label remaining data.csv",index=False)

comment_label1.to_csv("Comment 200 label data.csv",index=False)
comment_rest1.to_csv("Comment 200 label remaining data.csv",index=False)