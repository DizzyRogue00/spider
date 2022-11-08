class BilibiliApiException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class NoPermissionException(BilibiliApiException):
    def __init__(self, msg="无操作权限"):
        self.msg = msg


class BilibiliException(BilibiliApiException):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg

    def __str__(self):
        return "错误代码：%s, 信息：%s" % (self.code, self.msg)


class NetworkException(BilibiliApiException):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return "网络错误。状态码：%s" % self.code


class NoIdException(BilibiliApiException):
    def __init__(self):
        self.msg = "aid和bvid请至少提供一个"


class LiveException(BilibiliApiException):
    def __init__(self, msg: str):
        super().__init__(msg)

class UploadException(BilibiliApiException):
    def __init__(self, msg: str):
        super().__init__(msg)