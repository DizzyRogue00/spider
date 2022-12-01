import datetime
import json
import operator
import struct
import time
from functools import reduce

import requests

import exceptions
import utils
import random
import re
import pandas as pd
from BytesReader import BytesReader

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


# def get_danmaku_view(page_id: int, bvid: str = None, aid: int = None, verify: utils.Verify = None):
#     if verify is None:
#         verify = utils.Verify()
#     if not (aid or bvid):
#         raise exceptions.NoIdException
#     if not aid:
#         aid = utils.bvid2aid(bvid)
#     resp = requests.get(url_get_danmaku_view, params={
#         "type": 1,
#         "oid": page_id,
#         "pid": aid
#     }, headers=utils.DEFAULT_HEADERS, cookies=verify.get_cookies())
#     if resp.ok:
#         resp_data = resp.content
#         json_data = {}
#
#         '''
#         pos = 0
#         length = len(resp_data)
#         read_varint = utils.read_varint
#         '''
#         #2022/11/30 {
#         reader=BytesReader(resp_data)
#         #2022/11/30 }
#         '''
#         def read_dmSge(stream: bytes):
#             length_ = len(stream)
#             pos = 0
#             data = {}
#             while pos < length_:
#                 t = stream[pos] >> 3
#                 pos += 1
#                 if t == 1:
#                     d, l = read_varint(stream[pos:])
#                     data['pageSize'] = int(d)
#                     pos += l
#                 elif t == 2:
#                     d, l = read_varint(stream[pos:])
#                     data['total'] = int(d)
#                     pos += l
#                 else:
#                     continue
#             return data
#         '''
#         #2022/11/30 {
#         def read_dmSge(stream:bytes):
#             reader_=BytesReader(stream)
#             data={}
#             while not reader_.has_end():
#                 t=reader_.varint()>>3
#                 if t==1:
#                     data['page_size']=reader_.varint()
#                 elif t==2:
#                     data['total']=reader_.varint()
#                 else:
#                     continue
#             return data
#         #2022/11/30 }
#
#         '''
#         def read_flag(stream: bytes):
#             length_ = len(stream)
#             pos = 0
#             data = {}
#             while pos < length_:
#                 t = stream[pos] >> 3
#                 pos += 1
#                 if t == 1:
#                     d, l = read_varint(stream[pos:])
#                     data['recFlag'] = int(d)
#                     pos += l
#                 elif t == 2:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['recText'] = stream[pos:pos + str_len].decode()
#                     pos += str_len
#                 elif t == 3:
#                     d, l = read_varint(stream[pos:])
#                     data['recSwitch'] = int(d)
#                     pos += l
#                 else:
#                     continue
#             return data
#         '''
#
#         # 2022/11/30 {
#         def read_flag(stream: bytes):
#             reader_ = BytesReader(stream)
#             data = {}
#             while not reader_.has_end():
#                 t = reader_.varint() >> 3
#                 if t == 1:
#                     data['rec_flag'] = reader_.varint()
#                 elif t == 2:
#                     data['rec_text'] = reader_.string()
#                 elif t==3:
#                     data['rec_switch']=reader_.varint()
#                 else:
#                     continue
#             return data
#         # 2022/11/30 }
#
#         '''
#         def read_commandDms(stream: bytes):
#             length_ = len(stream)
#             pos = 0
#             data = {}
#             while pos < length_:
#                 t = stream[pos] >> 3
#                 pos += 1
#                 if t == 1:
#                     d, l = read_varint(stream[pos:])
#                     data['id'] = int(d)
#                     pos += l
#                 elif t == 2:
#                     d, l = read_varint(stream[pos:])
#                     data['oid'] = int(d)
#                     pos += l
#                 elif t == 3:
#                     d, l = read_varint(stream[pos:])
#                     data['mid'] = int(d)
#                     pos += l
#                 elif t == 4:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['commend'] = stream[pos: pos + str_len].decode()
#                     pos += str_len
#                 elif t == 5:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['content'] = stream[pos: pos + str_len].decode()
#                     pos += str_len
#                 elif t == 6:
#                     d, l = read_varint(stream[pos:])
#                     data['progress'] = int(d)
#                     pos += l
#                 elif t == 7:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['ctime'] = stream[pos: pos + str_len].decode()
#                     pos += str_len
#                 elif t == 8:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['mtime'] = stream[pos: pos + str_len].decode()
#                     pos += str_len
#                 elif t == 9:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['extra'] = json.loads(stream[pos: pos + str_len].decode())
#                     pos += str_len
#                 elif t == 10:
#                     str_len, l = read_varint(stream[pos:])
#                     pos += l
#                     data['idStr'] = stream[pos: pos + str_len].decode()
#                     pos += str_len
#                 else:
#                     continue
#             return data
#         '''
#         # 2022/11/30 {
#         def read_commandDms(stream: bytes):
#             reader_ = BytesReader(stream)
#             data = {}
#             while not reader_.has_end():
#                 t = reader_.varint() >> 3
#                 if t == 1:
#                     data['id'] = reader_.varint()
#                 elif t == 2:
#                     data['oid'] = reader_.varint()
#                 elif t==3:
#                     data['mid']=reader_.varint()
#                 elif t==4:
#                     data['commend']=reader_.string()
#                 elif t==5:
#                     data['content']=reader_.string()
#                 elif t==6:
#                     data['progress']=reader_.varint()
#                 elif t==7:
#                     data['ctime']=reader_.string()
#                 elif t==8:
#                     data['mtime']=reader_.string()
#                 elif t==9:
#                     print(reader_.string())
#                     #data['extra']=json.loads(reader_.string())
#                 elif t==10:
#                     data['id_str']=reader_.string()
#                 else:
#                     continue
#             return data
#         # 2022/11/30 }
#
#         # def read_dmSetting(stream: bytes):
#         #     length_ = len(stream)
#         #     pos = 0
#         #     data = {}
#         #     while pos < length_:
#         #         t = stream[pos] >> 3
#         #         pos += 1
#         #         if t == 1:
#         #             data['dmSwitch'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 2:
#         #             data['aiSwitch'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 3:
#         #             d, l = read_varint(stream[pos:])
#         #             data['aiLevel'] = int(d)
#         #             pos += l
#         #         elif t == 4:
#         #             data['blocktop'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 5:
#         #             data['blockscroll'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 6:
#         #             data['blockbottom'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 7:
#         #             data['blockcolor'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 8:
#         #             data['blockspecial'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 9:
#         #             data['preventshade'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 10:
#         #             data['dmask'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 11:
#         #             if len(stream[pos:pos + 4]) == 4:
#         #                 d = struct.unpack('>f', stream[pos: pos + 4])[0]
#         #                 pos += 4
#         #                 data['opacity'] = d
#         #             else:
#         #                 pos += len(stream[pos:pos + 4])
#         #                 continue
#         #         elif t == 12:
#         #             d, l = read_varint(stream[pos:])
#         #             data['dmarea'] = int(d)
#         #             pos += l
#         #         elif t == 13:
#         #             if len(stream[pos:pos + 4]) == 4:
#         #                 d = struct.unpack('>f', stream[pos: pos + 4])[0]
#         #                 pos += 4
#         #                 data['speedplus'] = d
#         #             else:
#         #                 pos += len(stream[pos:pos + 4])
#         #                 continue
#         #         elif t == 14:
#         #             if len(stream[pos:pos + 4]) == 4:
#         #                 d = struct.unpack('>f', stream[pos: pos + 4])[0]
#         #                 pos += 4
#         #                 data['fontsize'] = d
#         #             else:
#         #                 pos += len(stream[pos:pos + 4])
#         #                 continue
#         #         elif t == 15:
#         #             data['screensync'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 16:
#         #             data['speedsync'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 17:
#         #             str_len, l = read_varint(stream[pos:])
#         #             pos += l
#         #             data['fontfamily'] = stream[pos: pos + str_len].decode()
#         #             pos += str_len
#         #         elif t == 18:
#         #             data['bold'] = True if stream[pos] == b'\x01' else False
#         #             pos += 1
#         #         elif t == 19:
#         #             d, l = read_varint(stream[pos:])
#         #             data['fontborder'] = int(d)
#         #             pos += l
#         #         elif t == 20:
#         #             str_len, l = read_varint(stream[pos:])
#         #             pos += l
#         #             data['drawType'] = stream[pos: pos + str_len].decode()
#         #             pos += str_len
#         #         else:
#         #             continue
#         #     return data
#
#         #2022/11/30 {
#         def read_dmSetting(stream: bytes):
#             reader_ = BytesReader(stream)
#             data = {}
#             while not reader_.has_end():
#                 t = reader_.varint() >> 3
#
#                 if t == 1:
#                     data['dm_switch'] = reader_.bool()
#                 elif t == 2:
#                     data['ai_switch'] = reader_.bool()
#                 elif t == 3:
#                     data['ai_level'] = reader_.varint()
#                 elif t == 4:
#                     data['enable_top'] = reader_.bool()
#                 elif t == 5:
#                     data['enable_scroll'] = reader_.bool()
#                 elif t == 6:
#                     data['enable_bottom'] = reader_.bool()
#                 elif t == 7:
#                     data['enable_color'] = reader_.bool()
#                 elif t == 8:
#                     data['enable_special'] = reader_.bool()
#                 elif t == 9:
#                     data['prevent_shade'] = reader_.bool()
#                 elif t == 10:
#                     data['dmask'] = reader_.bool()
#                 elif t == 11:
#                     data['opacity'] = reader_.float(True)
#                 elif t == 12:
#                     data['dm_area'] = reader_.varint()
#                 elif t == 13:
#                     data['speed_plus'] = reader_.float(True)
#                 elif t == 14:
#                     data['font_size'] = reader_.float(True)
#                 elif t == 15:
#                     data['screen_sync'] = reader_.bool()
#                 elif t == 16:
#                     data['speed_sync'] = reader_.bool()
#                 elif t == 17:
#                     data['font_family'] = reader_.string()
#                 elif t == 18:
#                     data['bold'] = reader_.bool()
#                 elif t == 19:
#                     data['font_border'] = reader_.varint()
#                 elif t == 20:
#                     data['draw_type'] = reader_.string()
#                 else:
#                     continue
#             return data
#         #2022/11/30 }
#
#         # while pos < length:
#         #     type_ = resp_data[pos] >> 3
#         #     pos += 1
#         #     if type_ == 1:
#         #         d, l = read_varint(resp_data[pos:])
#         #         json_data['state'] = int(d)
#         #         pos += l
#         #     elif type_ == 2:
#         #         str_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['text'] = resp_data[pos:pos + str_len].decode()
#         #         pos += str_len
#         #     elif type_ == 3:
#         #         str_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['textSide'] = resp_data[pos:pos + str_len].decode()
#         #         pos += str_len
#         #     elif type_ == 4:
#         #         data_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['dmSge'] = read_dmSge(resp_data[pos:pos + data_len])
#         #         pos += data_len
#         #     elif type_ == 5:
#         #         data_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['flag'] = read_flag(resp_data[pos:pos + data_len])
#         #         pos += data_len
#         #     elif type_ == 6:
#         #         if 'specialDms' not in json_data:
#         #             json_data['specialDms'] = []
#         #         data_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['specialDms'].append(resp_data[pos: pos + data_len].decode())
#         #         pos += data_len
#         #     elif type_ == 7:
#         #         json_data['checkBox'] = True if resp_data[pos] == b'\x01' else False
#         #         pos += 1
#         #     elif type_ == 8:
#         #         d, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['count'] = int(d)
#         #     elif type_ == 9:
#         #         data_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         if 'commandDms' not in json_data:
#         #             json_data['commandDms'] = []
#         #         json_data['commandDms'].append(read_commandDms(resp_data[pos: pos + data_len]))
#         #         pos += data_len
#         #
#         #     elif type_ == 10:
#         #         data_len, l = read_varint(resp_data[pos:])
#         #         pos += l
#         #         json_data['dmSetting'] = read_dmSetting(resp_data[pos: pos + data_len])
#         #         pos += data_len
#         #     else:
#         #         continue
#         # return json_data
#
#         #2022/11/30 {
#         while not reader.has_end():
#             type_ = reader.varint() >> 3
#
#             if type_ == 1:
#                 json_data['state'] = reader.varint()
#             elif type_ == 2:
#                 json_data['text'] = reader.string()
#             elif type_ == 3:
#                 json_data['text_side'] = reader.string()
#             elif type_ == 4:
#                 json_data['dm_seg'] = read_dmSge(reader.bytes_string())
#             elif type_ == 5:
#                 json_data['flag'] = read_flag(reader.bytes_string())
#             elif type_ == 6:
#                 if 'special_dms' not in json_data:
#                     json_data['special_dms'] = []
#                 json_data['special_dms'].append(reader.string())
#             elif type_ == 7:
#                 json_data['check_box'] = reader.bool()
#             elif type_ == 8:
#                 json_data['count'] = reader.varint()
#             elif type_ == 9:
#                 if 'command_dms' not in json_data:
#                     json_data['command_dms'] = []
#                 json_data['command_dms'].append(
#                     read_commandDms(reader.bytes_string()))
#             elif type_ == 10:
#                 json_data['dm_setting'] = read_dmSetting(reader.bytes_string())
#             else:
#                 continue
#         return json_data
#         #2022/11/30 }
#     else:
#         raise exceptions.NetworkException(resp.status_code)
#
#
# def get_danmaku(bvid: str = None, aid: int = None, page_id: int = 0, verify: utils.Verify = None,
#                 date: datetime.date = None):
#     """
#     :param aid:
#     :param bvid:
#     :param page_id: 分p id，请先调用 get_video_info() ，先len(["pages"]),然后取其中的 ["pages"][分P号-1]["cid"]
#     :param verify: date不为None时需要SESSDATA验证
#     :param date: 为None时获取最新弹幕，为datetime.date时获取历史弹幕
#     """
#     dms = get_danmaku_g(bvid, aid, page_id, verify, date)
#     return dms
#
#
# def get_danmaku_g(bvid: str = None, aid: int = None, page_id: int = 0, verify: utils.Verify = None,
#                   date: datetime.date = None):
#     """
#     :param date: 为None时为最新数据，不为时为历史弹幕
#     :return:
#     """
#     if not (aid or bvid):
#         raise exceptions.NoIdException
#     if verify is None:
#         verify = utils.Verify()
#     if date is not None:
#         if not verify.has_sess():
#             raise exceptions.NoPermissionException
#     if not aid:
#         aid = utils.bvid2aid(bvid)
#     params = {
#         "oid": page_id,
#         "pid": aid,
#         "type": 1,
#         "segment_index": 1
#     }
#     if date is not None:
#         # params["date"]=date.strftime("%Y-%m-%d")
#         params = {
#             "oid": page_id,
#             "type": 1,
#             "date": date.strftime("%Y-%m-%d")
#         }
#
#     def parse_bdoc(url, params=None, cookies=None, headers=None):
#         if params is None:
#             params = {}
#         if cookies is None:
#             cookies = {}
#         if headers is None:
#             headers = utils.DEFAULT_HEADERS
#         req = requests.get(url, params=params, headers=headers, cookies=cookies)
#         if req.ok:
#             content_type = req.headers['content-type']
#             if content_type == 'application/json':
#                 con = req.json()
#                 if con['code'] != 0:
#                     raise exceptions.BilibiliException(con['code'], con['message'])
#                 else:
#                     return con
#
#             # elif content_type == 'application/octet-stream':
#             #     con = req.content
#             #     data = con
#             #     offset = 0
#             #
#             #     if data == b'\x10\x01':
#             #
#             #         raise exceptions.BilibiliApiException('视频弹幕已关闭')
#             #     while offset < len(data):
#             #         if data[offset] == 0x0a:
#             #             dm = utils.Danmaku('')
#             #             offset += 1
#             #             dm_data_length, l = utils.read_varint(data[offset:])
#             #             offset += l
#             #             real_data = data[offset:offset + dm_data_length]
#             #             dm_data_offset = 0
#             #             while dm_data_offset < dm_data_length:
#             #                 data_type = real_data[dm_data_offset] >> 3
#             #                 dm_data_offset += 1
#             #                 if data_type == 1:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.id = d
#             #                 elif data_type == 2:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.dm_time = datetime.timedelta(seconds=d / 1000)
#             #                 elif data_type == 3:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.mode = d
#             #                 elif data_type == 4:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.font_size = d
#             #                 elif data_type == 5:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.color = utils.Color()
#             #                     dm.color.set_dec_color(d)
#             #                 elif data_type == 6:
#             #                     str_len = real_data[dm_data_offset]
#             #                     dm_data_offset += 1
#             #                     d = real_data[dm_data_offset:dm_data_offset + str_len]
#             #                     dm_data_offset += str_len
#             #                     dm.crc32_id = d.decode(errors='ignore')
#             #                     # dm.crack_uid()
#             #                     # dm.uid=dm.crack_uid()
#             #                     # dm.uid1,dm.uid2=dm.crack_uid()
#             #                 elif data_type == 7:
#             #                     str_len = real_data[dm_data_offset]
#             #                     dm_data_offset += 1
#             #                     d = real_data[dm_data_offset:dm_data_offset + str_len]
#             #                     dm_data_offset += str_len
#             #                     dm.text = d.decode(errors='ignore')
#             #                 elif data_type == 8:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.send_time = datetime.datetime.fromtimestamp(d)
#             #                 elif data_type == 9:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.weight = d
#             #                 elif data_type == 10:
#             #                     str_len = real_data[dm_data_offset]
#             #                     dm_data_offset += 1
#             #                     d = real_data[dm_data_offset:dm_data_offset + str_len]
#             #                     dm_data_offset += str_len
#             #                     dm.action = d.decode(errors='ignore')
#             #                 elif data_type == 11:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.pool = d
#             #                 elif data_type == 12:
#             #                     str_len = real_data[dm_data_offset]
#             #                     dm_data_offset += 1
#             #                     d = real_data[dm_data_offset:dm_data_offset + str_len]
#             #                     dm_data_offset += str_len
#             #                     dm.id_str = d.decode(errors='ignore')
#             #                 elif data_type == 13:
#             #                     d, l = utils.read_varint(real_data[dm_data_offset:])
#             #                     dm_data_offset += l
#             #                     dm.attr = d
#             #                 else:
#             #                     break
#             #             offset += dm_data_length
#             #             yield dm
#
#             #2022/11/30 {
#             elif content_type == 'application/octet-stream':
#                 con = req.content
#                 data = con
#                 if data ==b'\x10\x01':
#                     raise exceptions.BilibiliApiException('视频弹幕已关闭')
#
#                 reader=BytesReader(data)
#                 while not reader.has_end():
#                     dm = utils.Danmaku('')
#                     dm_pack_data = reader.bytes_string()
#                     dm_reader = BytesReader(dm_pack_data)
#                     while not dm_reader.has_end():
#                         data_type=dm_reader.varint()>>3
#                         if data_type == 1:
#                             dm.id = dm_reader.varint()
#                         elif data_type == 2:
#                             dm.dm_time = dm_reader.varint() / 1000
#                         elif data_type == 3:
#                             dm.mode = dm_reader.varint()
#                         elif data_type == 4:
#                             dm.font_size = dm_reader.varint()
#                         elif data_type == 5:
#                             dm.color = hex(dm_reader.varint())[2:]
#                         elif data_type == 6:
#                             dm.crc32_id = dm_reader.string()
#                         elif data_type == 7:
#                             dm.text = dm_reader.string()
#                         elif data_type == 8:
#                             dm.send_time = dm_reader.varint()
#                         elif data_type == 9:
#                             dm.weight = dm_reader.varint()
#                         elif data_type == 10:
#                             dm.action = dm_reader.varint()
#                         elif data_type == 11:
#                             dm.pool = dm_reader.varint()
#                         elif data_type == 12:
#                             dm.id_str = dm_reader.string()
#                         elif data_type == 13:
#                             dm.attr = dm_reader.varint()
#                         else:
#                             break
#                     yield dm
#             #2022/11/30 }
#         else:
#             raise exceptions.NetworkException(req.status_code)
#
#     if date is None:
#         view = get_danmaku_view(page_id, aid=aid)
#         seg_count = view['dm_seg']['total']
#         danmakus = []
#         for i in range(seg_count):
#             params['segment_index'] = i + 1
#             dms = parse_bdoc(url_get_danmaku, params=params, cookies=verify.get_cookies(),
#                              headers=utils.DEFAULT_HEADERS)
#             for d in dms:
#                 danmakus.append(d)
#
#     else:
#         danmakus = []
#         dms = parse_bdoc(url_get_history_danmaku, params=params, cookies=verify.get_cookies(),
#                          headers=utils.DEFAULT_HEADERS)
#         for d in dms:
#             danmakus.append(d)
#     return danmakus


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

def get_danmaku_date(bvid: str = None, aid: int = None, page_id: int = 0, verify: utils.Verify = None,
                date: datetime.date = None):
    """
    :param aid:
    :param bvid:
    :param page_id: 分p id，请先调用 get_video_info() ，先len(["pages"]),然后取其中的 ["pages"][分P号-1]["cid"]
    :param verify: date不为None时需要SESSDATA验证
    :param date: 为None时获取最新弹幕，为datetime.date时获取历史弹幕
    """
    if not (aid or bvid):
        raise exceptions.NoIdException
    if date is not None:
        # params["date"]=date.strftime("%Y-%m-%d")
        params = {
            "oid": page_id,
            "type": 1,
            "date": date.strftime("%Y-%m-%d")
        }
        if not verify.has_sess():
            raise exceptions.NoPermissionException
    if not aid:
        aid = utils.bvid2aid(bvid)
    req=requests.get(url_get_history_danmaku,params=params,headers=utils.DEFAULT_HEADERS,cookies=verify.get_cookies())
    def dm_construct(iter,date):
        new_iter=iter[0:]
        dm = utils.Danmaku('')
        dm.send_time = date.strftime("%Y-%m-%d")
        dm.text=new_iter
        return dm
    if req.ok:
        req_list=re.findall(':.(.*?)@',req.text)
        dms=[dm_construct(i,date) for i in req_list]
    else:
        raise exceptions.NetworkException(req.status_code)
    return dms

class Get_Danmaku_history(object):
    def __init__(self, bvid, verify):
        self.bvid = bvid
        self.info = None
        self.verify = verify
        self.danmaku_index = None
        self.latest_danmaku = None
        self.page_id = None
        self.history_danmaku = None
        self.date = None
        self.output = None

    def get_info(self):
        self.info = get_video_info(self.bvid, 1)
        return self.info

    def get_index(self):
        self.danmaku_index = get_history_danmaku_index(bvid=self.bvid, verify=self.verify)
        return self.danmaku_index

    def get_history_danmaku(self, page_id, date):
        self.page_id = page_id
        self.date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        history_danmaku = get_danmaku_date(bvid=self.bvid, page_id=self.page_id, verify=self.verify, date=self.date)
        self.history_danmaku = [str(x) for x in history_danmaku]
        return self.history_danmaku

    def get_total_danmaku(self):
        danmaku_index = self.get_index()
        info = self.get_info()
        page_len = len(info["pages"])
        for i in range(page_len):
            data = []
            for j in danmaku_index['pages'][i]:
                data_new = self.get_history_danmaku(info["pages"][i]["cid"], j)
                data.append(data_new)
                time.sleep(random.randint(1, 3))
            #print(data)
            temp_output = reduce(operator.add, data)
            print(len(temp_output))
            #temp_output1 = set(temp_output)
            #output = list(temp_output1)
            #output.sort()
            self.output = temp_output
            print(len(self.output))
            # print(output[625:670])
            data = pd.DataFrame([i.split(',', 1) for i in self.output],
                                columns=['send_time',  'text'])
            # print(data)
            name = self.bvid + '_' + str(info["pages"][i]["cid"]) + '.csv'
            data.to_csv(name, index=False, sep=',')
    def get_danmaku_overview(self):
        info=self.get_info()
        page_len = len(info["pages"])
        def dm_generate(iter):
            dm=utils.Danmaku("")
            dm_message=iter[0].split(",")
            dm.dm_time=datetime.timedelta(seconds=float(dm_message[0]))
            dm.mode=int(dm_message[1])
            dm.font_size=int(dm_message[2])
            dm.send_time=datetime.datetime.fromtimestamp(float(dm_message[4]))
            dm.crc32_uid=str(dm_message[6])
            #dm.uid=dm.crack_uid()
            dm.text=iter[1]
            #print(dm)
            return dm

        for i in range(page_len):
            cid=str(info["pages"][i]["cid"])
            url_get_danmaku_overview = "https://comment.bilibili.com/%s.xml" % cid
            response=requests.get(url_get_danmaku_overview,headers=utils.DEFAULT_HEADERS,cookies=self.verify.get_cookies())
            response.encoding="utf-8"
            text=response.text
            data=re.findall('<d p="(.*?)">(.*?)</d>',text)
            danmakus=[str(dm_generate(j)) for j in data]
            #print(danmakus[0].split(',',1))
            #temp_output = reduce(operator.add, danmakus)
            self.output = danmakus
            print(len(self.output))
            data = pd.DataFrame([i.split(',', 1) for i in self.output],
                                columns=['send_time',  'text'])
            # print(data)
            name = self.bvid + '_' + str(info["pages"][i]["cid"])+'overview' + '.csv'
            data.to_csv(name, index=False, sep=',')
    def __len__(self):
        return len(self.output)
