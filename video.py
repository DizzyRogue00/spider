import datetime
import json
import operator
import struct
import time
from functools import reduce

import requests

import exceptions
import utils

url_simple = "https://api.bilibili.com/x/web-interface/archive/stat"
url_detail = "https://api.bilibili.com/x/web-interface/view"
url_index = "https://api.bilibili.com/x/v2/dm/history/index"
url_get_danmaku = "https://api.bilibili.com/x/v2/dm/web/seg.so"
url_get_history_danmaku = "https://api.bilibili.com/x/v2/dm/web/history/seg.so"
url_get_danmaku_view = "https://api.bilibili.com/x/v2/dm/web/view"


def get_video_info(bvid: str = None, aid: int = None, verify: utils.Verify = None, option=1):
    if not (aid or bvid):
        raise exceptions.NoIdException
    if verify is None:
        verify = utils.Verify()
    params = {
        "aid": aid,
        "bvid": bvid
    }
    if option == 1:
        info = utils.get(url_detail, params=params, cookies=verify.get_cookies())
    else:
        info = utils.get(url_simple, params=params, cookies=verify.get_cookies())
    return info


def get_danmaku_view(page_id: int, bvid: str = None, aid: int = None, verify: utils.Verify = None):
    if verify is None:
        verify = utils.Verify()
    if not (aid or bvid):
        raise exceptions.NoIdException
    if not aid:
        aid = utils.bvid2aid(bvid)
    resp = requests.get(url_get_danmaku_view, params={
        "type": 1,
        "oid": page_id,
        "pid": aid
    }, headers=utils.DEFAULT_HEADERS, cookies=verify.get_cookies())
    if resp.ok:
        resp_data = resp.content
        json_data = {}
        pos = 0
        length = len(resp_data)
        read_varint = utils.read_varint

        def read_dmSge(stream: bytes):
            length_ = len(stream)
            pos = 0
            data = {}
            while pos < length_:
                t = stream[pos] >> 3
                pos += 1
                if t == 1:
                    d, l = read_varint(stream[pos:])
                    data['pageSize'] = int(d)
                    pos += l
                elif t == 2:
                    d, l = read_varint(stream[pos:])
                    data['total'] = int(d)
                    pos += l
                else:
                    continue
            return data

        def read_flag(stream: bytes):
            length_ = len(stream)
            pos = 0
            data = {}
            while pos < length_:
                t = stream[pos] >> 3
                pos += 1
                if t == 1:
                    d, l = read_varint(stream[pos:])
                    data['recFlag'] = int(d)
                    pos += l
                elif t == 2:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['recText'] = stream[pos:pos + str_len].decode()
                    pos += str_len
                elif t == 3:
                    d, l = read_varint(stream[pos:])
                    data['recSwitch'] = int(d)
                    pos += l
                else:
                    continue
            return data

        def read_commandDms(stream: bytes):
            length_ = len(stream)
            pos = 0
            data = {}
            while pos < length_:
                t = stream[pos] >> 3
                pos += 1
                if t == 1:
                    d, l = read_varint(stream[pos:])
                    data['id'] = int(d)
                    pos += l
                elif t == 2:
                    d, l = read_varint(stream[pos:])
                    data['oid'] = int(d)
                    pos += l
                elif t == 3:
                    d, l = read_varint(stream[pos:])
                    data['mid'] = int(d)
                    pos += l
                elif t == 4:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['commend'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                elif t == 5:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['content'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                elif t == 6:
                    d, l = read_varint(stream[pos:])
                    data['progress'] = int(d)
                    pos += l
                elif t == 7:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['ctime'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                elif t == 8:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['mtime'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                elif t == 9:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['extra'] = json.loads(stream[pos: pos + str_len].decode())
                    pos += str_len
                elif t == 10:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['idStr'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                else:
                    continue
            return data

        def read_dmSetting(stream: bytes):
            length_ = len(stream)
            pos = 0
            data = {}
            while pos < length_:
                t = stream[pos] >> 3
                pos += 1
                if t == 1:
                    data['dmSwitch'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 2:
                    data['aiSwitch'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 3:
                    d, l = read_varint(stream[pos:])
                    data['aiLevel'] = int(d)
                    pos += l
                elif t == 4:
                    data['blocktop'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 5:
                    data['blockscroll'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 6:
                    data['blockbottom'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 7:
                    data['blockcolor'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 8:
                    data['blockspecial'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 9:
                    data['preventshade'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 10:
                    data['dmask'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 11:
                    if len(stream[pos:pos + 4]) == 4:
                        d = struct.unpack('>f', stream[pos: pos + 4])[0]
                        pos += 4
                        data['opacity'] = d
                    else:
                        pos += len(stream[pos:pos + 4])
                        continue
                elif t == 12:
                    d, l = read_varint(stream[pos:])
                    data['dmarea'] = int(d)
                    pos += l
                elif t == 13:
                    if len(stream[pos:pos + 4]) == 4:
                        d = struct.unpack('>f', stream[pos: pos + 4])[0]
                        pos += 4
                        data['speedplus'] = d
                    else:
                        pos += len(stream[pos:pos + 4])
                        continue
                elif t == 14:
                    if len(stream[pos:pos + 4]) == 4:
                        d = struct.unpack('>f', stream[pos: pos + 4])[0]
                        pos += 4
                        data['fontsize'] = d
                    else:
                        pos += len(stream[pos:pos + 4])
                        continue
                elif t == 15:
                    data['screensync'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 16:
                    data['speedsync'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 17:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['fontfamily'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                elif t == 18:
                    data['bold'] = True if stream[pos] == b'\x01' else False
                    pos += 1
                elif t == 19:
                    d, l = read_varint(stream[pos:])
                    data['fontborder'] = int(d)
                    pos += l
                elif t == 20:
                    str_len, l = read_varint(stream[pos:])
                    pos += l
                    data['drawType'] = stream[pos: pos + str_len].decode()
                    pos += str_len
                else:
                    continue
            return data

        while pos < length:
            type_ = resp_data[pos] >> 3
            pos += 1
            if type_ == 1:
                d, l = read_varint(resp_data[pos:])
                json_data['state'] = int(d)
                pos += l
            elif type_ == 2:
                str_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['text'] = resp_data[pos:pos + str_len].decode()
                pos += str_len
            elif type_ == 3:
                str_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['textSide'] = resp_data[pos:pos + str_len].decode()
                pos += str_len
            elif type_ == 4:
                data_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['dmSge'] = read_dmSge(resp_data[pos:pos + data_len])
                pos += data_len
            elif type_ == 5:
                data_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['flag'] = read_flag(resp_data[pos:pos + data_len])
                pos += data_len
            elif type_ == 6:
                if 'specialDms' not in json_data:
                    json_data['specialDms'] = []
                data_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['specialDms'].append(resp_data[pos: pos + data_len].decode())
                pos += data_len
            elif type_ == 7:
                json_data['checkBox'] = True if resp_data[pos] == b'\x01' else False
                pos += 1
            elif type_ == 8:
                d, l = read_varint(resp_data[pos:])
                pos += l
                json_data['count'] = int(d)
            elif type_ == 9:
                data_len, l = read_varint(resp_data[pos:])
                pos += l
                if 'commandDms' not in json_data:
                    json_data['commandDms'] = []
                json_data['commandDms'].append(read_commandDms(resp_data[pos: pos + data_len]))
                pos += data_len

            elif type_ == 10:
                data_len, l = read_varint(resp_data[pos:])
                pos += l
                json_data['dmSetting'] = read_dmSetting(resp_data[pos: pos + data_len])
                pos += data_len
            else:
                continue
        return json_data

    else:
        raise exceptions.NetworkException(resp.status_code)


def get_danmaku(bvid: str = None, aid: int = None, page_id: int = 0, verify: utils.Verify = None,
                date: datetime.date = None):
    """
    :param aid:
    :param bvid:
    :param page_id: 分p id，请先调用 get_video_info() ，先len(["pages"]),然后取其中的 ["pages"][分P号-1]["cid"]
    :param verify: date不为None时需要SESSDATA验证
    :param date: 为None时获取最新弹幕，为datetime.date时获取历史弹幕
    """
    dms = get_danmaku_g(bvid, aid, page_id, verify, date)
    return dms


def get_danmaku_g(bvid: str = None, aid: int = None, page_id: int = 0, verify: utils.Verify = None,
                  date: datetime.date = None):
    """
    :param date: 为None时为最新数据，不为时为历史弹幕
    :return:
    """
    if not (aid or bvid):
        raise exceptions.NoIdException
    if verify is None:
        verify = utils.Verify()
    if date is not None:
        if not verify.has_sess():
            raise exceptions.NoPermissionException
    if not aid:
        aid = utils.bvid2aid(bvid)
    params = {
        "oid": page_id,
        "pid": aid,
        "type": 1,
        "segment_index": 1
    }
    if date is not None:
        # params["date"]=date.strftime("%Y-%m-%d")
        params = {
            "oid": page_id,
            "type": 1,
            "date": date.strftime("%Y-%m-%d")
        }

    def parse_bdoc(url, params=None, cookies=None, headers=None):
        if params is None:
            params = {}
        if cookies is None:
            cookies = {}
        if headers is None:
            headers = utils.DEFAULT_HEADERS
        req = requests.get(url, params=params, headers=headers, cookies=cookies)
        if req.ok:
            content_type = req.headers['content-type']
            if content_type == 'application/json':
                con = req.json()
                if con['code'] != 0:
                    raise exceptions.BilibiliException(con['code'], con['message'])
                else:
                    return con
            elif content_type == 'application/octet-stream':
                con = req.content
                data = con
                offset = 0
                if data == b'\x10\x01':
                    raise exceptions.BilibiliApiException('视频弹幕已关闭')
                while offset < len(data):
                    if data[offset] == 0x0a:
                        dm = utils.Danmaku('')
                        offset += 1
                        dm_data_length, l = utils.read_varint(data[offset:])
                        offset += l
                        real_data = data[offset:offset + dm_data_length]
                        dm_data_offset = 0
                        while dm_data_offset < dm_data_length:
                            data_type = real_data[dm_data_offset] >> 3
                            dm_data_offset += 1
                            if data_type == 1:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.id = d
                            elif data_type == 2:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.dm_time = datetime.timedelta(seconds=d / 1000)
                            elif data_type == 3:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.mode = d
                            elif data_type == 4:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.font_size = d
                            elif data_type == 5:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.color = utils.Color()
                                dm.color.set_dec_color(d)
                            elif data_type == 6:
                                str_len = real_data[dm_data_offset]
                                dm_data_offset += 1
                                d = real_data[dm_data_offset:dm_data_offset + str_len]
                                dm_data_offset += str_len
                                dm.crc32_id = d.decode(errors='ignore')
                                # dm.crack_uid()
                                # dm.uid=dm.crack_uid()
                                # dm.uid1,dm.uid2=dm.crack_uid()
                            elif data_type == 7:
                                str_len = real_data[dm_data_offset]
                                dm_data_offset += 1
                                d = real_data[dm_data_offset:dm_data_offset + str_len]
                                dm_data_offset += str_len
                                dm.text = d.decode(errors='ignore')
                            elif data_type == 8:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.send_time = datetime.datetime.fromtimestamp(d)
                            elif data_type == 9:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.weight = d
                            elif data_type == 10:
                                str_len = real_data[dm_data_offset]
                                dm_data_offset += 1
                                d = real_data[dm_data_offset:dm_data_offset + str_len]
                                dm_data_offset += str_len
                                dm.action = d.decode(errors='ignore')
                            elif data_type == 11:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.pool = d
                            elif data_type == 12:
                                str_len = real_data[dm_data_offset]
                                dm_data_offset += 1
                                d = real_data[dm_data_offset:dm_data_offset + str_len]
                                dm_data_offset += str_len
                                dm.id_str = d.decode(errors='ignore')
                            elif data_type == 13:
                                d, l = utils.read_varint(real_data[dm_data_offset:])
                                dm_data_offset += l
                                dm.attr = d
                            else:
                                break
                        offset += dm_data_length
                        yield dm
        else:
            raise exceptions.NetworkException(req.status_code)

    if date is None:
        view = get_danmaku_view(page_id, aid=aid)
        seg_count = view['dmSge']['total']
        danmakus = []
        for i in range(seg_count):
            params['segment_index'] = i + 1
            dms = parse_bdoc(url_get_danmaku, params=params, cookies=verify.get_cookies(),
                             headers=utils.DEFAULT_HEADERS)
            for d in dms:
                danmakus.append(d)

    else:
        danmakus = []
        dms = parse_bdoc(url_get_history_danmaku, params=params, cookies=verify.get_cookies(),
                         headers=utils.DEFAULT_HEADERS)
        for d in dms:
            danmakus.append(d)
    return danmakus


def dateRange(beginDate, endDate):
    dates = []
    start = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    end = datetime.datetime.strptime(endDate, "%Y-%m-%d")
    while start <= end:
        date = datetime.datetime.strftime(start, "%Y-%m-%d")
        dates.append(date)
        start = start + datetime.timedelta(days=1)
    return dates


def monthRange(beginDate, endDate):
    monthSet = set()
    for date in dateRange(beginDate, endDate):
        monthSet.add(date[0:7])
    monthlist = []
    for month in monthSet:
        monthlist.append(month)
    return sorted(monthlist)


# def get_history_danmaku_index(bvid:str=None,aid:int=None,page:int=0,verify:utils.Verify=None):
def get_history_danmaku_index(bvid: str = None, aid: int = None, verify: utils.Verify = None):
    if not (aid or bvid):
        raise exceptions.NoIdException
    if verify is None:
        verify = utils.Verify()
    if not verify.has_sess():
        raise exceptions.NoPermissionException
    info = get_video_info(aid=aid, bvid=bvid, verify=verify)
    index = {}
    index["pages"] = []
    page_len = len(info["pages"])
    dateEnd = datetime.date.fromtimestamp(time.time())
    dateStart = datetime.date.fromtimestamp(info["pubdate"])
    ds = datetime.datetime.strftime(dateStart, "%Y-%m-%d")
    de = datetime.datetime.strftime(dateEnd, "%Y-%m-%d")
    monthlist = monthRange(ds, de)
    for i in range(page_len):
        page_id = info["pages"][i]["cid"]
        date_list = []
        for j in monthlist:
            params = {
                "oid": page_id,
                "month": j,
                "type": 1
            }
            get = utils.get(url=url_index, params=params, cookies=verify.get_cookies())
            date_list.append(get)
        # print(date_list)
        date_list = list(filter(None, date_list))
        output = reduce(operator.add, date_list)
        index["pages"].append(output)
    return index
