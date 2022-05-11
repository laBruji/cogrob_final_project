#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dispatcher class to help with online scheduling
"""

import re
import time
import numpy as np

class Dispatcher():
    def __init__(self, disturbance_max_delay=0.0, sim_time=False, quiet=False):
        self.disturbance_max_delay = disturbance_max_delay
        self.sim_time = sim_time
        self.quiet = quiet

        self.time_start = None
        self.t = None
        self.noisy_events = set(['C', 'E'])
        self.execution_trace = {}

    def start(self):
        if self.sim_time:
            self.time_start = 0.0
            self.t = 0.0
        else:
            self.time_start = time.time()

        if not self.quiet:
            self.print_time_message("Starting", 0.0)
        return self.time_start

    def done(self):
        # if not self.quiet:
        #     self.print_time_message("Done!", self.time())
        pass

    def time(self):
        if self.sim_time:
            return self.t
        else:
            return time.time() - self.time_start

    def sleep(self, amount):
        if amount < 0:
            raise Exception("Called sleep for a negative amount of time!")

        if self.sim_time:
            self.t += amount
        else:
            time.sleep(amount)

    def dispatch(self, event):
        if self.time_start is None:
            raise Exception("You need to call dispatcher.start() before you dispatch anything.")
        if event in self.execution_trace:
            raise Exception("Trying to dispatch event {}, but it was already dispatched earlier!".format(event))

        if event in self.noisy_events:
            delay = self.simulate_disturbance()
        else:
            delay = 0.0
        t = self.time()
        self.execution_trace[event] = t
        if not self.quiet:
            self.print_time_message("Dispatched {} {}".format(
                                        event, ("" if delay == 0.0 else "(delay {:0.4f}s)".format(delay))),
                                    t)
        return t
    
    def simulate_disturbance(self):
        delay = np.random.random() * self.disturbance_max_delay
        self.sleep(delay)
        return delay

    def print_time_message(self, message, time):
        print("\033[33m{:08.4f}\033[0m: {}".format(time, message))

