
import threading


class ModelThread(threading.Thread):
    """
    处理task相关的线程类
    """

    def __init__(self, func, args):
        super(ModelThread, self).__init__()
        self.func = func  # 要执行的task类型
        self.args = args  # 要传入的参数
        self.result = None

    def run(self):
        # 线程类实例调用start()方法将执行run()方法,这里定义具体要做的异步任务
        print("start func {}".format(self.func.__name__))  # 打印task名字　用方法名.__name__
        try:
            print(self.args)
            self.result = self.func(self.args)  # 将任务执行结果赋值给self.result变量
        except Exception as e:
            print(e)

    def join(self, timeout=None):
        super(ModelThread, self).join()
        return self.result


def generate(input):
    return input['input']


a = dict(input="123123123")
print(generate(a))
a = ModelThread(func=generate, args=a)
a.start()
result = a.join()
print(str(result))

