from inspect import getfullargspec
class Function(object):
    def __init__(self,fn):
        self.fn=fn

    def __call__(self,*args,**kwargs):
        fn=Namespace.get_instance().get(self.fn,*args)
        if not fn:
            raise Exception('no matching function found')
        return fn(*args,**kwargs)

    def key(self,args=None):
        if args is None:
            args=getfullargspec(self.fn).args
        return tuple(
            [
                self.fn.__module__,
                self.fn.__class__,
                self.fn.__name__,
                len(args or []),
            ]
        )

class Namespace(object):
    _instance=None
    def __init__(self):
        if self._instance is None:
            self.function_map=dict()
            Namespace._instance=self
        else:
            raise Exception("cannot instantiate a virtual Namespace again")

    @staticmethod
    def get_instance():
        if Namespace._instance is None:
            Namespace()
        return Namespace._instance

    def register(self,fn):
        func=Function(fn)
        self.function_map[func.key()]=fn
        return func

    def get(self,fn,*args):
        func=Function(fn)
        return self.function_map.get(func.key(args=args))

def overload(fn):
        return Namespace.get_instance().register(fn)
