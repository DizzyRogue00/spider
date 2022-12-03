'''
#pip install bilibili-api
# 导入模块
from bilibili_api import video
# 参数
BVID = "BV1jK4y1D7Ft"
# 获取视频信息
info = video.get_video_info(bvid=BVID)
# 假设这里获取 p1 的最新弹幕信息，需要取出 page_id，即每 p 都有自己的编号
page_id = info["pages"][0]["cid"]
# 然后开始获取弹幕
danmakus = video.get_danmaku(bvid=BVID, page_id=page_id)
# 打印出来！
for dm in danmakus:
    print(str(dm))
'''

'''
!!!
关键信息，不要泄露
buvid3=046C7BCB-9307-4073-8CCE-2FFA0AFE38DD185002infoc; 
buvid_fp_plain=046C7BCB-9307-4073-8CCE-2FFA0AFE38DD185002infoc; 
buvid_fp=046C7BCB-9307-4073-8CCE-2FFA0AFE38DD185002infoc; 
SESSDATA=fdfe071a%2C1633418816%2C94f66%2A41; 
bili_jct=c8da65a998aabc369b3e29a19277b868; 
'''

import datetime
import operator
import random
import time
from functools import reduce

import pandas as pd

import comments
import video
from utils import Verify

# url_simple="https://api.bilibili.com/x/web-interface/archive/stat"
# url_detail="https://api.bilibili.com/x/web-interface/view"
'''
#Final
start_time=datetime.datetime.now()
'''
# BVID = "BV1jK4y1D7Ft"
# BVID="BV1ob411p7oc"#这个视频分p
'''
#Final
BVID="BV1KK4y1N7xT"
info=video.get_video_info(BVID,1)
print(info)
'''
# print(len(info["pages"]))
'''
#Final
verify=Verify("fdfe071a%2C1633418816%2C94f66%2A41","c8da65a998aabc369b3e29a19277b868")
'''
# print(verify)
'''
danmaku_index=video.get_history_danmaku_index(bvid=BVID,verify=verify)
'''
# print(danmaku_index)
# print(danmaku_index['pages'])
# print(danmaku_index['pages'][0][1])
'''
ZC=video.get_danmaku_view(page_id=info["pages"][0]["cid"],bvid=BVID,verify=verify)
'''
# ZC=video.get_danmaku_view(page_id=info["pages"][0]["cid"],aid=929491224)
# print(ZC)
# danmakus=video.get_danmaku(bvid=BVID, page_id=info["pages"][0]["cid"],verify=verify,date=datetime.date(2021,5,7))
'''
danmakus=video.get_danmaku(bvid=BVID, page_id=info["pages"][0]["cid"],verify=verify)

for dm in danmakus:
    print(str)
    '''
'''
data=[]
danmaku_index=video.get_history_danmaku_index(bvid=BVID,verify=verify)
print(danmaku_index['pages'][0][0])
print(type(danmaku_index['pages'][0][0]))
j_date=datetime.datetime.strptime(danmaku_index['pages'][0][0],'%Y-%m-%d').date()
danmakus = video.get_danmaku(bvid=BVID, page_id=info["pages"][0]["cid"], verify=verify,date=j_date)
print(type(danmakus))
data_new=[str(j) for x in danmakus for j in x]
data.append(data_new)
print(len(data_new))
data=reduce(operator.add,data)
print(len(data))
for i in data:
    print(i)
'''
'''
#Final
danmaku_index=video.get_history_danmaku_index(bvid=BVID,verify=verify)
page_len=len(info["pages"])
for i in range(page_len):
    data=[]
    danmakus = video.get_danmaku(bvid=BVID, page_id=info["pages"][i]["cid"], verify=verify)
    data_new=[str(x) for x in danmakus]
    data.append(data_new)
    time.sleep(random.randint(1, 3))
    for j in danmaku_index['pages'][i]:
        j_date=datetime.datetime.strptime(j,'%Y-%m-%d').date()
        print(j_date)
        danmakus = video.get_danmaku(bvid=BVID, page_id=info["pages"][i]["cid"], verify=verify,date=j_date)
        data_new=[str(x) for x in danmakus]
        data.append(data_new)
        time.sleep(random.randint(1, 3))
    temp_output=reduce(operator.add,data)
    print(len(temp_output))
    temp_output1=set(temp_output)
    output=list(temp_output1)
    output.sort()
    print(output[0])
    print(len(output))
    #print(output[625:670])
    data=pd.DataFrame([i.split(',',3) for i in output], columns=['send_time','dm_time','crc32_id','text'])
    #print(data)
    name=BVID+'_'+str(info["pages"][i]["cid"])+'.csv'
    data.to_csv(name,index=False,sep=',')
delta = (datetime.datetime.now() - start_time).total_seconds()

print(f'用时：{delta}s')
'''
def Main(BVID,verify):
    start_time = datetime.datetime.now()
    #BVID = "BV1KK4y1N7xT"
    #BVID = "BV16A411G7Sd"
    #verify = Verify("fdfe071a%2C1633418816%2C94f66%2A41", "c8da65a998aabc369b3e29a19277b868")
    #verify = Verify("f6d5fe36%2C1644567166%2Ce3f14%2A81", "fbfd8b641c81de76f13b6232039833ec")
    #instance = Get_Danmaku(BVID, verify)
    '''
    #heuristic method
    instance = video.Get_Danmaku_heuristic(BVID, verify)
    info = instance.get_info()
    print(info)
    #instance.get_total_danmaku()
    instance.get_danmaku_overview()
    delta = (datetime.datetime.now() - start_time).total_seconds()
    print(f'用时：{delta}s')
    '''
    #exact method
    instance = video.Get_Danmaku(BVID, verify)
    info = instance.get_info()
    print(info)
    # print(instance.get_index())
    #latest_danmaku=instance.get_latest_danmaku(info["pages"][0]["cid"])
    #print(latest_danmaku)
    #print(len(latest_danmaku))
    # print(instance.get_history_danmaku(info["pages"][0]["cid"],'2021-05-09'))
    instance.get_total_danmaku()
    delta = (datetime.datetime.now() - start_time).total_seconds()
    print(f'用时：{delta}s')


if __name__ == '__main__':
    #BVID = "BV1jK4y1D7Ft"
    #BVID="BV1ob411p7oc"#这个视频分p
    #BVID = "BV1KK4y1N7xT"
    BVID = "BV1wm4y1D7vr"
    #verify = Verify("fdfe071a%2C1633418816%2C94f66%2A41", "c8da65a998aabc369b3e29a19277b868")
    verify = Verify("eb20d1fa%2C1685607731%2C6a5be%2Ac1", "16bd458fe5352259a6ba70977d18c692")
    #爬取弹幕
    Main(BVID,verify)

    '''
    #原始的爬取评论，速度慢
    comments.getComments(BVID=BVID, verify=verify)
    '''

    #利用多线程爬取展示评论
    #comments.getOriginalComments(BVID=BVID, verify=verify)
    '''
    #两个版本，全部的评论，休眠600s
    time.sleep(600)

    #time.sleep(600)

    #利用线程爬取总评论
    comments.getTotalComments(BVID=BVID, verify=verify)

    time.sleep(600)

    #利用线程爬取总评论版本2
    comments.getTotalComments_ver2(BVID=BVID, verify=verify)
    '''





