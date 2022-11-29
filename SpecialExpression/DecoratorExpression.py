import time

def runtime(func):
    def get_time():
        print(time.time())
        func()
    return get_time

@runtime
def student_run():
    print("run")

student_run()