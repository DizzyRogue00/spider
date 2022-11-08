import datetime
import math
import random
import time
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

import utils

url_get_comments = "https://api.bilibili.com/x/v2/reply/main"
url_get_sub_comments = "https://api.bilibili.com/x/v2/reply/reply"


class Comments(object):
    def __init__(self, text: str, send_time: float = time.time(), uname: str = None, mid: str = None, level: int = 0,
                 sex: str = "保密", like: int = 0, is_sub: bool = False):
        self.text = text
        self.send_time = datetime.datetime.fromtimestamp(send_time)

        self.uname = uname
        self.mid = mid
        self.level = level
        self.sex = sex

        self.like = like
        self.is_sub = is_sub

    def __str__(self):
        ret = "{0},{1},{2},{3},{4},{5},{6},{7}".format(self.uname, self.mid, self.level, self.sex, self.send_time,
                                                       self.like, self.is_sub, self.text)
        return ret

    def __len__(self):
        return len(self.text)


class Node(object):
    def __init__(self, Obj, lchild=None, rchild=None):
        self.Obj = Obj
        self.Objchild = []
        self.lchild = lchild
        self.rchild = rchild


class Tree(object):
    def __init__(self, root=None):
        self.root = root

    def add(self, Obj):
        node = Node(Obj)
        if self.root == None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            while queue:
                cur = queue.pop(0)
                if cur.lchild == None:
                    cur.lchild = node
                    return
                elif cur.rchild == None:
                    cur.rchild = node
                    return
                else:
                    queue.append(cur.lchild)
                    queue.append(cur.rchild)

    def breadth_traverse(self):
        if self.root == None:
            return
        queue = []
        final_queue = []
        queue.append(self.root)
        final_queue.append([self.root.Obj.Obj])
        final_queue.append(self.root.Obj.Objchild)
        while queue:
            node = queue.pop(0)
            if node.lchild != None:
                queue.append(node.lchild)
                final_queue.append([node.lchild.Obj.Obj])
                final_queue.append(node.lchild.Obj.Objchild)
            if node.rchild != None:
                queue.append(node.rchild)
                final_queue.append([node.rchild.Obj.Obj])
                final_queue.append(node.rchild.Obj.Objchild)
        temp = list(filter(None, final_queue))
        temp_queue = list(chain(*temp))
        return temp_queue


def get_total_num(oid: int, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": 0,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    return resp["cursor"]["all_count"]


def get_comments_raw(oid: int, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": 0,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    return resp


def get_comments(oid: int, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": 0,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments_tr = Tree()
    while True:
        resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
        if "replies" not in resp:
            break
        if resp["replies"] is None:
            break
        num = len(resp["replies"])
        for i in range(num):
            time.sleep(random.randint(1, 2))
            comment = Comments("")
            comment.text = resp["replies"][i]["content"]["message"]
            comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
            comment.uname = resp["replies"][i]["member"]["uname"]
            comment.mid = resp["replies"][i]["member"]["mid"]
            comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
            comment.sex = resp["replies"][i]["member"]["sex"]
            comment.like = resp["replies"][i]["like"]
            # comment=str(comment)
            comment_node = Node(comment)
            if "replies" in resp["replies"][i]:
                if resp["replies"][i]["replies"] is not None:
                    num_ = len(resp["replies"][i]["replies"])
                    for j in range(num_):
                        comment_sub = Comments("")
                        comment_sub.text = resp["replies"][i]["replies"][j]["content"]["message"]
                        comment_sub.send_time = datetime.datetime.fromtimestamp(
                            resp["replies"][i]["replies"][j]["ctime"])
                        comment_sub.uname = resp["replies"][i]["replies"][j]["member"]["uname"]
                        comment_sub.mid = resp["replies"][i]["replies"][j]["member"]["mid"]
                        comment_sub.level = resp["replies"][i]["replies"][j]["member"]["level_info"]["current_level"]
                        comment_sub.sex = resp["replies"][i]["replies"][j]["member"]["sex"]
                        comment_sub.like = resp["replies"][i]["replies"][j]["like"]
                        comment_sub.is_sub = True
                        # comment_sub=str(comment_sub)
                        comment_node.Objchild.append(comment_sub)
            comments_tr.add(comment_node)
        next = resp["cursor"]["next"]
        params["next"] = next
    comments = comments_tr.breadth_traverse()
    return comments


def get_sub_comments_raw(oid: int, root: int, ps: int = 10, pn: int = 1, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "pn": pn,
        "type": 1,
        "ps": ps,
        "root": root
    }
    resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    count = resp["page"]["count"]
    size = resp["page"]["size"]
    resp_comments = []
    for i in range(math.ceil(count / size)):
        time.sleep(random.randint(1, 2))
        if "replies" in resp:
            if resp["replies"] is not None:
                num = len(resp["replies"])
                for j in range(num):
                    comment = Comments("")
                    comment.text = resp["replies"][j]["content"]["message"]
                    comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][j]["ctime"])
                    comment.uname = resp["replies"][j]["member"]["uname"]
                    comment.mid = resp["replies"][j]["member"]["mid"]
                    comment.level = resp["replies"][j]["member"]["level_info"]["current_level"]
                    comment.sex = resp["replies"][j]["member"]["sex"]
                    comment.like = resp["replies"][j]["like"]
                    comment.is_sub = True
                    resp_comments.append(comment)
        pn += 1
        if pn > math.ceil(count / size):
            break
        params["pn"] = pn
        resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    return resp_comments


def get_total_comments(oid: int, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": 0,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments_tr = Tree()
    while True:
        resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
        if "replies" not in resp:
            break
        if resp["replies"] is None:
            break
        num = len(resp["replies"])
        for i in range(num):
            time.sleep(random.randint(1, 2))
            comment = Comments("")
            comment.text = resp["replies"][i]["content"]["message"]
            comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
            comment.uname = resp["replies"][i]["member"]["uname"]
            comment.mid = resp["replies"][i]["member"]["mid"]
            comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
            comment.sex = resp["replies"][i]["member"]["sex"]
            comment.like = resp["replies"][i]["like"]
            # comment=str(comment)
            comment_node = Node(comment)
            if "replies" in resp["replies"][i]:
                if resp["replies"][i]["replies"] is not None:
                    rpid = resp["replies"][i]["rpid"]
                    comment_sub = get_sub_comments_raw(oid=oid, root=rpid, verify=verify)
                    comment_node.Objchild.append(comment_sub)
                    comment_node.Objchild = list(chain(*comment_node.Objchild))
            comments_tr.add(comment_node)
        next = resp["cursor"]["next"]
        params["next"] = next
    comments = comments_tr.breadth_traverse()
    return comments

def get_Max_page(oid:int,verify:utils.Verify=None):
    if verify is None:
        verify=utils.Verify()
    params={
        "oid":oid,
        "next":0,
        "type":1,
        "mode":3,
        "plat":1
    }
    resp=utils.get(url_get_comments,params,cookies=verify.get_cookies())
    max_page=math.ceil(resp["cursor"]["all_count"]/20)
    return max_page

def get_eff_comments(oid: int, next:int=0,verify: utils.Verify = None):
    time.sleep(random.randint(1, 10))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": next,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments=[]
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    if "replies" in resp:
        if resp["replies"] is not None:
            num = len(resp["replies"])
            for i in range(num):
                #time.sleep(random.randint(1, 3))
                comment = Comments("")
                comment.text = resp["replies"][i]["content"]["message"]
                comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
                comment.uname = resp["replies"][i]["member"]["uname"]
                comment.mid = resp["replies"][i]["member"]["mid"]
                comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
                comment.sex = resp["replies"][i]["member"]["sex"]
                comment.like = resp["replies"][i]["like"]
                comments.append(comment)
                if "replies" in resp["replies"][i]:
                    if resp["replies"][i]["replies"] is not None:
                        num_ = len(resp["replies"][i]["replies"])
                        for j in range(num_):
                            comment_sub = Comments("")
                            comment_sub.text = resp["replies"][i]["replies"][j]["content"]["message"]
                            comment_sub.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["replies"][j]["ctime"])
                            comment_sub.uname = resp["replies"][i]["replies"][j]["member"]["uname"]
                            comment_sub.mid = resp["replies"][i]["replies"][j]["member"]["mid"]
                            comment_sub.level = resp["replies"][i]["replies"][j]["member"]["level_info"]["current_level"]
                            comment_sub.sex = resp["replies"][i]["replies"][j]["member"]["sex"]
                            comment_sub.like = resp["replies"][i]["replies"][j]["like"]
                            comment_sub.is_sub = True
                            comments.append(comment_sub)
    return comments

def get_sub_comments_page(oid:int,root:int,ps:int=10,pn:int=1,verify:utils.Verify=None):
    if verify is None:
        verify=utils.Verify()
    params = {
        "oid": oid,
        "pn": pn,
        "type": 1,
        "ps": ps,
        "root": root
    }
    resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    count = resp["page"]["count"]
    size = resp["page"]["size"]
    page_size=math.ceil(count / size)
    return page_size

def get_eff_sub_comments_raw(oid: int, root: int, ps: int = 10, pn: int = 1, verify: utils.Verify = None):
    time.sleep(random.randint(1,10))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "pn": pn,
        "type": 1,
        "ps": ps,
        "root": root
    }
    resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    resp_comments=[]
    if "replies" in resp:
        if resp["replies"] is not None:
            num = len(resp["replies"])
            for j in range(num):
                comment = Comments("")
                comment.text = resp["replies"][j]["content"]["message"]
                comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][j]["ctime"])
                comment.uname = resp["replies"][j]["member"]["uname"]
                comment.mid = resp["replies"][j]["member"]["mid"]
                comment.level = resp["replies"][j]["member"]["level_info"]["current_level"]
                comment.sex = resp["replies"][j]["member"]["sex"]
                comment.like = resp["replies"][j]["like"]
                comment.is_sub = True
                resp_comments.append(comment)
    return resp_comments
'''
def get_eff_total_comments(oid: int, next:int=0,verify: utils.Verify = None):
    time.sleep(random.randint(1, 2))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": next,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments=[]
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    if "replies" in resp:
        if resp["replies"] is not None:
            num = len(resp["replies"])
            for i in range(num):
                time.sleep(1)
                comment = Comments("")
                comment.text = resp["replies"][i]["content"]["message"]
                comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
                comment.uname = resp["replies"][i]["member"]["uname"]
                comment.mid = resp["replies"][i]["member"]["mid"]
                comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
                comment.sex = resp["replies"][i]["member"]["sex"]
                comment.like = resp["replies"][i]["like"]
                comments.append([comment])
                if "replies" in resp["replies"][i]:
                    if resp["replies"][i]["replies"] is not None:
                        rpid = resp["replies"][i]["rpid"]
                        page_size=get_sub_comments_page(oid=oid, root=rpid, verify=verify)
                        with ThreadPoolExecutor(max_workers=4) as t:
                            obj_list=[]
                            for page in range(1,page_size+1):
                                obj=t.submit(get_eff_sub_comments_raw,oid, rpid, 10, page, verify)
                                obj_list.append(obj)
                            for future in as_completed(obj_list):
                                data=future.result()
                                comments.append(data)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    return comments
'''

def get_eff_total_comments(oid: int, next:int=0,verify: utils.Verify = None):
    time.sleep(random.randint(1, 10))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": next,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments=[]
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    if "replies" in resp:
        if resp["replies"] is not None:
            num = len(resp["replies"])
            for i in range(num):
                comment = Comments("")
                comment.text = resp["replies"][i]["content"]["message"]
                comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
                comment.uname = resp["replies"][i]["member"]["uname"]
                comment.mid = resp["replies"][i]["member"]["mid"]
                comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
                comment.sex = resp["replies"][i]["member"]["sex"]
                comment.like = resp["replies"][i]["like"]
                comments.append([comment])
                if "replies" in resp["replies"][i]:
                    if resp["replies"][i]["replies"] is not None:
                        rpid = resp["replies"][i]["rpid"]
                        page_size=get_sub_comments_page(oid=oid, root=rpid, verify=verify)
                        time.sleep(random.randint(1,3))
                        oid_=[oid]*page_size
                        rpid_=[rpid]*page_size
                        ps_=[10]*page_size
                        page=list(range(1,page_size+1))
                        verify_=[verify]*page_size
                        with ThreadPoolExecutor(max_workers=2) as t:
                            for data in t.map(get_eff_sub_comments_raw,oid_,rpid_,ps_,page,verify_):
                                comments.append(data)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    return comments

def get_sub_comments_raw_ver2(oid: int, root: int, ps: int = 10, pn: int = 1, verify: utils.Verify = None):
    time.sleep(random.randint(1,10))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "pn": pn,
        "type": 1,
        "ps": ps,
        "root": root
    }
    resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    count = resp["page"]["count"]
    size = resp["page"]["size"]
    resp_comments = []
    for i in range(math.ceil(count / size)):
        time.sleep(random.randint(1, 2))
        if "replies" in resp:
            if resp["replies"] is not None:
                num = len(resp["replies"])
                for j in range(num):
                    comment = Comments("")
                    comment.text = resp["replies"][j]["content"]["message"]
                    comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][j]["ctime"])
                    comment.uname = resp["replies"][j]["member"]["uname"]
                    comment.mid = resp["replies"][j]["member"]["mid"]
                    comment.level = resp["replies"][j]["member"]["level_info"]["current_level"]
                    comment.sex = resp["replies"][j]["member"]["sex"]
                    comment.like = resp["replies"][j]["like"]
                    comment.is_sub = True
                    resp_comments.append(comment)
        pn += 1
        if pn > math.ceil(count / size):
            break
        params["pn"] = pn
        resp = utils.get(url_get_sub_comments, params=params, cookies=verify.get_cookies())
    return resp_comments

def get_eff_total_comments_ver2(oid: int, next:int=0,verify: utils.Verify = None):
    time.sleep(random.randint(1, 20))
    if verify is None:
        verify = utils.Verify()
    params = {
        "oid": oid,
        "next": next,
        "type": 1,
        "mode": 3,
        "plat": 1
    }
    comments=[]
    resp = utils.get(url_get_comments, params=params, cookies=verify.get_cookies())
    if "replies" in resp:
        if resp["replies"] is not None:
            num = len(resp["replies"])
            for i in range(num):
                comment = Comments("")
                comment.text = resp["replies"][i]["content"]["message"]
                comment.send_time = datetime.datetime.fromtimestamp(resp["replies"][i]["ctime"])
                comment.uname = resp["replies"][i]["member"]["uname"]
                comment.mid = resp["replies"][i]["member"]["mid"]
                comment.level = resp["replies"][i]["member"]["level_info"]["current_level"]
                comment.sex = resp["replies"][i]["member"]["sex"]
                comment.like = resp["replies"][i]["like"]
                comments.append([comment])
                if "replies" in resp["replies"][i]:
                    if resp["replies"][i]["replies"] is not None:
                        rpid = resp["replies"][i]["rpid"]
                        comment_sub = get_sub_comments_raw(oid=oid, root=rpid, verify=verify)
                        comments.append(comment_sub)
    comments=list(filter(None,comments))
    comments=list(chain(*comments))
    return comments