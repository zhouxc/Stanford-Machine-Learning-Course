#!/usr/bin/env python

"""
tftask.py -- programmtic utilities for designating problem set tasks.
"""

import StringIO
import sys

import eventlog

def _capture_stdout(fxn,*args,**kwargs):
    fileOldStdout = sys.stdout
    sys.stdout = StringIO.StringIO()
    try:
        oRet = fxn(*args, **kwargs)
        sOut = sys.stdout.getvalue()
        sys.stdout.close()
    finally:
        sys.stdout = fileOldStdout
    return sOut,oRet

class BaseTask(object):
    _IS_TASK = True
    def get_name(self):
        return None
    def get_description(self):
        return None
    def dependencies(self):
        return None
    def get_type(self):
        return None
    def get_priority(self):
        return 0
    def validate(self, oOut):
        return None
    def task(self):
        raise TypeError("%s has no task." % self.__class__.__name__)
    def run(self):
        fValidation = True
        tb = None
        sConsole = None
        oOut = None
        try:
            sConsole,oOut = _capture_stdout(self.task)
            fValidation = self.validate(oOut)
        except:
            import traceback
            fValidation = False
            tb = traceback.format_exc()
        if fValidation is not False:
            eventlog.task_success(self.get_name(), 0)
        else:
            eventlog.task_failure(self.get_name(), 0, tb)
        return {"console": sConsole, "result": oOut, "valid": fValidation,
                "tb": tb}

class GraphTask(BaseTask):
    def get_type(self):
        return "graph"
    def validate(self, listPair):
        for tpl in listPair:
            if len(tpl) < 2:
                return False
        return True

class ChartTask(BaseTask):
    def get_type(self):
        return "chart"
    

def list_tasks(mod):
    if isinstance(mod,basestring):
        mod = globals()[mod]
    listTask = []
    for sName in dir(mod):
        o = getattr(mod,sName)
        if isinstance(o,type) and hasattr(o,"_IS_TASK"):
            listTask.append(o())
    return listTask
