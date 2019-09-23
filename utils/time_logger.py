import time

class time_logger():
    def __init__(self, time_logger_step, hierachy = 1):
        if time_logger_step == 0:
            self.logQ = False
        else:
            self.logQ = True
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s):
        if self.logQ and (self.step_count%self.time_logger_step==0):
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f"%(time.time() - self.time))
            self.time = time.time()

    def update(self):
        self.step_count += 1
        if self.logQ:
            self.log("#Refresh logger")
            self.newline()


    def newline(self):
        if self.logQ:
            print('\n')
