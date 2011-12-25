#!/usr/bin/env python

"""
monitortests.py -- keep track of passing and failing tests
"""

import inspect
import unittest
import StringIO
import sys

import eventlog
import loadconfig

class Fixture(object):
    def __init__(self, listSt, mod, modWork):
        self.mod = mod
        self.modWork = modWork
        self.testDict = {}
        for st in listSt:
            self.testDict[st.sName] = st
    def get(self,sName):
        return self.testDict.get(sName)
    def serialize(self):
        listSt = self.testDict.values()
        listSt.sort(lambda a,b: a.cLine - b.cLine)
        return [st.serialize() for st in listSt]
    def run_multiple(self, listSName):
        reload(self.modWork)
        listOut = []
        for sName in listSName:
            listOut.append({"name":sName, "results": self.get(sName).run()})
        return listOut
            
class SingleTest(object):
    def __init__(self, tsParent, test):
        self.sSuiteName = self.extract_suite_name(tsParent)
        self.sMethodName = self.extract_method_name(test)
        self.sName = "%s.%s" % (self.sSuiteName,  self.sMethodName)
        self.test = test
        self.nResult = None
        self.cLine = self.get_line_number()
    @classmethod
    def extract_suite_name(cls,tsParent):
        return tsParent.__class__.__name__
    @classmethod
    def extract_method_name(cls,test):
        return test._testMethodName
    def run(self):
        tr = unittest.TestResult()
        fileOldStdout = sys.stdout
        sys.stdout = StringIO.StringIO()
        sConsole = None
        try:
            self.test.run(tr)
            sConsole = sys.stdout.getvalue()
        finally:
            sys.stdout.close()
            sys.stdout = fileOldStdout
        if tr.errors or tr.failures:
            sTb = str((map(lambda (a,b): b, tr.errors + tr.failures))[0])
            eventlog.test_failure(self.sName, 0, sTb)
        else:
            eventlog.test_success(self.sName, 0)
        return serialize_test_result(tr,sConsole)
    def description(self):
        return self.test.__doc__ or "(no description)"
    def serialize(self):
        return {"name": self.sName, "description": self.description(),
                "result": self.nResult}
    def get_line_number(self):
        fxn = getattr(self.test, self.sMethodName)
        if hasattr(fxn,"wrapped"):
            fxn = fxn.wrapped
        return inspect.getsourcelines(fxn)[1]

def load_tests(mod,modWork):
    tl = unittest.TestLoader()
    listSt = []
    for ts in tl.loadTestsFromModule(mod):
        for test in ts._tests:
            listSt.append(SingleTest(ts,test))
    return Fixture(listSt,mod,modWork)

def serialize_test_result(tr,sConsole):
    listTb = map(lambda (a,b): b, tr.errors + tr.failures)
    return {"failures": listTb, "result": not listTb, "console": sConsole}

if __name__ == "__main__":
    import sys
    sys.path.append(loadconfig.get_config_dir())
    mod = __import__("testdtree")
    fxt = load_tests(mod)
    print fxt.run_multiple(["TestSuite.test_separate_by_attribute"])
