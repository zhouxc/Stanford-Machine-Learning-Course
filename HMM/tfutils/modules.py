#!/usr/bin/env python

"""
modules.py -- list the modules exported by tfutils
"""

TFUTILS_FILES = ("serveui.py", "tftask.py", "monitortests.py", "loadconfig.py",
                 "eventlog.py", "updatemanager.py", "modules.py",
                 "__init__.py", "static", "simplejson", "versions/origin.js")

if __name__ == "__main__":
    for sFile in TFUTILS_FILES:
        print sFile
