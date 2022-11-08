import datetime

import pandas as pd

import common
import utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
import time
import random

# BVID = "BV1jK4y1D7Ft"
# BVID="BV1yV411E74k"
'''
BVID="BV16A411G7Sd"
verify = Verify("fdfe071a%2C1633418816%2C94f66%2A41", "c8da65a998aabc369b3e29a19277b868")
oid=utils.bvid2aid(bvid=BVID)
ZC=common.get_comments_raw(oid=oid,verify=verify)
print(len(ZC["replies"]))
print(ZC)
print(ZC["replies"][0]["content"]["message"])
print(ZC["replies"][1])
print(ZC["replies"][0]["ctime"])
#print(ZC["data"]["replies"])
comments=common.get_comments(oid=oid,verify=verify)
#print(verify.get_cookies())
total_comments=common.get_total_comments(oid=oid,verify=verify)
total_num=common.get_total_num(oid=oid,verify=verify)
print(total_num)
for i in range(len(total_comments)):
    print(total_comments[i])
'''
'''
for i in range(len(comments)):
    print(comments[i])
'''


def getComments(BVID: str, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    start_time = datetime.datetime.now()
    oid = utils.bvid2aid(bvid=BVID)

    comments = common.get_comments(oid=oid, verify=verify)
    data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
    name_comments = BVID + '_comments' + '.csv'
    data_comments.to_csv(name_comments, index=False, sep=',')
    delta_comments = (datetime.datetime.now() - start_time).total_seconds()
    print(f'展示评论用时:{delta_comments}s')

    total_comments = common.get_total_comments(oid=oid, verify=verify)
    total_num = common.get_total_num(oid=oid, verify=verify)
    if len(total_comments) == total_num:
        print("Correct")
    elif len(total_comments) < total_num:
        print("There may be something wrong!")
    else:
        print("???")
    print(len(total_comments))
    data_total_comments = pd.DataFrame([str(i).split(',', 7) for i in total_comments],
                                       columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
    name_total_comments = BVID + '_total_comments' + '.csv'
    data_total_comments.to_csv(name_total_comments, index=False, sep=',')
    #delta_total_comments = (datetime.datetime.now() - start_time).total_seconds() - delta_comments
    delta = (datetime.datetime.now() - start_time).total_seconds()
    #print(f'总评论用时:{delta_total_comments}s')
    print(f'总用时:{delta}s')
'''
def getOriginalComments(BVID:int,verify:utils.Verify=None):
    if verify is None:
        verify=utils.Verify()
    oid = utils.bvid2aid(bvid=BVID)
    comments = []
    max_next=common.get_Max_page(oid=oid, verify=verify)+1
    with ThreadPoolExecutor(max_workers=4) as t:
        obj_list=[]
        start_time = datetime.datetime.now()
        for next in range(1,max_next):
            obj=t.submit(common.get_eff_comments,oid,next,verify)
            obj_list.append(obj)
        for future in as_completed(obj_list):
            data=future.result()
            comments.append(data)
        comments=list(filter(None,comments))
        comments=list(chain(*comments))
        data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
        name_comments = BVID + '_efficient_comments' + '.csv'
        data_comments.to_csv(name_comments, index=False, sep=',')
        delta_comments = (datetime.datetime.now() - start_time).total_seconds()
        print(f'展示评论用时:{delta_comments}s')
'''

def getOriginalComments(BVID:int,verify:utils.Verify=None):
    start_time = datetime.datetime.now()
    if verify is None:
        verify=utils.Verify()
    oid = utils.bvid2aid(bvid=BVID)
    comments = []
    max_next=common.get_Max_page(oid=oid, verify=verify)
    time.sleep(random.randint(1,3))
    oid_=[oid]*max_next
    next_=list(range(1,max_next+1))
    verify_=[verify]*max_next
    with ThreadPoolExecutor(max_workers=2) as t:
        for data in t.map(common.get_eff_comments,oid_,next_,verify_):
            comments.append(data)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    print(len(comments))
    data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
    name_comments = BVID + '_efficient_comments' + '.csv'
    data_comments.to_csv(name_comments, index=False, sep=',')
    delta_comments = (datetime.datetime.now() - start_time).total_seconds()
    print(f'展示评论用时:{delta_comments}s')

'''
def getTotalComments(BVID:int,verify:utils.Verify=None):
    if verify is None:
        verify=utils.Verify()
    oid = utils.bvid2aid(bvid=BVID)
    max_next=common.get_Max_page(oid=oid, verify=verify)+1
    comments = []
    with ThreadPoolExecutor(max_workers=4) as t:
        obj_list=[]
        start_time = datetime.datetime.now()
        for next in range(1,max_next):
            obj=t.submit(common.get_eff_total_comments,oid,next,verify)
            obj_list.append(obj)
        for future in as_completed(obj_list):
            data=future.result()
            comments.append(data)
        comments=list(filter(None,comments))
        comments=list(chain(*comments))
        total_num = common.get_total_num(oid=oid, verify=verify)
        if len(comments) == total_num:
            print("Correct")
        elif len(comments) < total_num:
            print("There may be something wrong!")
        else:
            print("???")
        print(len(comments))
        data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
        name_comments = BVID + '_efficient_total_comments' + '.csv'
        data_comments.to_csv(name_comments, index=False, sep=',')
        delta = (datetime.datetime.now() - start_time).total_seconds()
        print(f'总评论用时:{delta}s')
'''

def getTotalComments(BVID:int,verify:utils.Verify=None):
    start_time = datetime.datetime.now()
    if verify is None:
        verify=utils.Verify()
    oid = utils.bvid2aid(bvid=BVID)
    max_next=common.get_Max_page(oid=oid, verify=verify)
    time.sleep(random.randint(1,3))
    comments = []
    oid_=[oid]*max_next
    next_=list(range(1,max_next+1))
    verify_=[verify]*max_next
    with ThreadPoolExecutor(max_workers=2) as executor:
        for data in executor.map(common.get_eff_total_comments,oid_,next_,verify_):
        #for data in executor.map(common.get_eff_total_comments_ver2, oid_, next_, verify_):
            comments.append(data)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    total_num = common.get_total_num(oid=oid, verify=verify)
    if len(comments) == total_num:
        print("Correct")
    elif len(comments) < total_num:
        print("There may be something wrong!")
    else:
        print("???")
    print(len(comments))
    data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
    name_comments = BVID + '_efficient_total_comments' + '.csv'
    data_comments.to_csv(name_comments, index=False, sep=',')
    delta = (datetime.datetime.now() - start_time).total_seconds()
    print(f'总评论用时:{delta}s')

def getTotalComments_ver2(BVID:int,verify:utils.Verify=None):
    start_time = datetime.datetime.now()
    if verify is None:
        verify=utils.Verify()
    oid = utils.bvid2aid(bvid=BVID)
    max_next=common.get_Max_page(oid=oid, verify=verify)
    time.sleep(random.randint(1,3))
    comments = []
    oid_=[oid]*max_next
    next_=list(range(1,max_next+1))
    verify_=[verify]*max_next
    with ThreadPoolExecutor(max_workers=2) as executor:
        #for data in executor.map(common.get_eff_total_comments,oid_,next_,verify_):
        for data in executor.map(common.get_eff_total_comments_ver2, oid_, next_, verify_):
            comments.append(data)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    total_num = common.get_total_num(oid=oid, verify=verify)
    if len(comments) == total_num:
        print("Correct")
    elif len(comments) < total_num:
        print("There may be something wrong!")
    else:
        print("???")
    print(len(comments))
    data_comments = pd.DataFrame([str(i).split(',', 7) for i in comments],
                                 columns=['uname', 'mid', 'level', 'sex', 'send_time', 'like', 'is_sub', 'text'])
    name_comments = BVID + '_efficient_total_comments_ver2' + '.csv'
    data_comments.to_csv(name_comments, index=False, sep=',')
    delta = (datetime.datetime.now() - start_time).total_seconds()
    print(f'总评论用时:{delta}s')
