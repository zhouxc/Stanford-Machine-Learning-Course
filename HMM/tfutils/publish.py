#!/usr/bin/env python

"""
publish.py -- create a problem set from a Python module.
"""

import inspect
import re
import sys

DOCTEST_MOD = """
if __name__ == "__main__":
    import doctest
    doctest.testmod()"""

MAIN_FUNCTION = """
if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))"""

class CodeObject(object):
    def __init__(self, sIn, sOut, iLine):
        self.sIn = sIn
        self.sOut = sOut
        self.iLine = iLine
    @classmethod
    def from_module(cls, sKey, mod):
        if sKey == mod.__name__:
            sOut = "import %s" % sKey
        else:
            sOut = "import %s as %s" % (mod.__name__, sKey)
        return CodeObject("",sOut + "\n", inspect.getsourcelines(mod)[1])

def build_strip_re(sDoc):
    return re.compile(r'\s*""".*"""', re.DOTALL)

def strip_function(fxn,sSrc=None):
    sDoc = fxn.__doc__
    if sDoc is None:
        return
    reDoc = build_strip_re(sDoc)
    if sSrc is None:
        sSrc = inspect.getsource(fxn)
    listS = reDoc.split(sSrc, 1)
    return ('%s\n\t"""%s"""\n\traise NotImplementedError\n'
            % (listS[0],sDoc)).expandtabs(4)

def is_builtin(o):
    try:
        inspect.getfile(o)
    except TypeError:
        return True
    return False

def get_objects(mod):
    listCo = []
    for sName,o in inspect.getmembers(mod):
        if sName == "__doc__" and o is not None:
            listCo.append(CodeObject(o,'"""%s"""\n' % o,-2))
            continue
        if is_builtin(o) or hasattr(o,"ignore"):
            continue
        if inspect.ismodule(o):
            listCo.append(CodeObject.from_module(sName,o))
            continue
        listSrcLines,iLine = inspect.getsourcelines(o)
        sSrc = "".join(listSrcLines)
        if inspect.isfunction(o):
            if o.__doc__ is not None and not hasattr(o, "is_support"):
                sSrcOut = strip_function(o,sSrc)
            else:
                sSrcOut = sSrc
            listCo.append(CodeObject(sSrc, sSrcOut, iLine))
        else:
            listCo.append(CodeObject(sSrc, sSrc, iLine))
    return listCo

def dump_code_objects(listCo, fAddMain):
    listCo.sort(lambda a,b: a.iLine - b.iLine)
    listS = ["#!/usr/bin/env python\n\n"]
    for co in listCo:
        listS.append(co.sOut)
        listS.append("\n")
    if fAddMain:
        listS.extend(("\n", MAIN_FUNCTION, "\n"))
    return "".join(listS)

def clean_module(mod):
    return dump_code_objects(get_objects(mod), True)
        
def main(argv):
    import loadconfig
    sConfDir = loadconfig.get_config_dir()
    sys.path.append(sConfDir)
    dictConf = loadconfig.load_config_file(sConfDir)
    mod = __import__(dictConf["workmodule"].rsplit('.',1)[0]) #import dtree
    print dump_code_objects(get_objects(mod), True)

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
