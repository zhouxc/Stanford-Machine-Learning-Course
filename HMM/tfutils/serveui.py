#!/usr/bin/env python

"""
serveui.py - spawn an HTTP server listening on localhost at a high port
in order to provide UI functionality for problem set tasks.
"""

import BaseHTTPServer
try:
    import json
except ImportError:
    import simplejson as json
import optparse
import os
from os import path
import re
import signal
import sys
import threading
import urllib
import webbrowser

import eventlog
import loadconfig
import monitortests
import tftask
import updatemanager

DEFAULT_PORT = 14512
STATIC_CONTENT_PREFIX = "static"

STATIC_RE = re.compile(r'^[/]' + STATIC_CONTENT_PREFIX
                       + r'[/](?P<sStaticPath>.*)$')
TEST_RE = re.compile(r'^[/]test/(?P<sCommand>[^/]*)[/]$')
METADATA_RE = re.compile(r'^[/]metadata[/]$')
TASK_RE = re.compile(r'^[/]task/(?P<sTask>[^/]*)[/]$')
UPDATES_RE = re.compile(r'^[/]updates/(?P<sUpdateTask>[^/]*)[/]$')

GLOBAL_STATE = {}

CONFIG_DIR = loadconfig.get_config_dir()

def _static(*args):
    sSuffix = path.join(*args)
    if path.isabs(sSuffix):
        raise ValueError("Cannot serve an absolute path.")
    return path.join('/'+STATIC_CONTENT_PREFIX,sSuffix)

dictAlias = {"/": _static("index.html")}

def load_module(sPath):
    sPref,sSuff = path.splitext(sPath)
    if sSuff != ".py":
        raise ValueError("Invalid module: %s" % sPath)
    sAbsPath = path.join(CONFIG_DIR,path.basename(sPath))
    if sAbsPath not in sys.path:
        sys.path.append(path.dirname(sAbsPath))
    return __import__(path.basename(sPref))

def get_post_data(req):
    s = req.rfile.read(int(req.headers["content-length"]))
    listKv = s.split('&')
    listPairs = [sKv.split('=') for sKv in listKv]
    dictData = {}
    for sK,sV in listPairs:
        dictData[urllib.unquote(sK.lstrip('?'))] = urllib.unquote(sV)
    return dictData

def task_id(task):
    return task.get_name().lower().replace(" ","_")

def get_task_dict():
    if "modTask" not in GLOBAL_STATE:
        raise ValueError("No task module in global state.")
    mod = GLOBAL_STATE["modTask"]
    listTask = tftask.list_tasks(mod)
    return dict([(task_id(task), task) for task in listTask])

def get_task_metadata():
    dictTask = get_task_dict()
    listTaskData = []
    for task in dictTask.values():
        dictData = {"name": task.get_name(), "type": task.get_type(),
                    "description": task.get_description(),
                    "id":task_id(task), "priority": task.get_priority()}
        listTaskData.append(dictData)
    return listTaskData

def serve_static(req, sStaticPath):
    sPath = sStaticPath.lstrip('/')
    sStaticContent = GLOBAL_STATE["sStatic"]
    sFullPath = path.join(sStaticContent,sPath)
    if not path.isfile(sFullPath):
        raise ValueError("Not a file: %s" % sFullPath)
    try:
        infile = open(sFullPath, 'rb')
        sContents = infile.read()
    finally:
        infile.close()
    return sContents

def serve_metadata(req):
    dictConfig = GLOBAL_STATE["dictConfig"]
    return json.dumps({"sTaskTitle":dictConfig["title"],
                       "sTaskSubtitle":dictConfig["subtitle"],
                       "listTask": get_task_metadata()})

def serve_test(req, sCommand):
    if sCommand == "load":
        modTest = GLOBAL_STATE["modTest"]
        modWork = GLOBAL_STATE["modWork"]
        reload(modTest)
        reload(modWork)
        fxt = monitortests.load_tests(modTest,modWork)
        GLOBAL_STATE["fxt"] = fxt
        return json.dumps(fxt.serialize())
    if sCommand == "run" and req.command.lower() == "post":
        if "fxt" not in GLOBAL_STATE:
            raise ValueError("No fixture loaded.")
        fxt = GLOBAL_STATE["fxt"]
        listToRun = get_post_data(req)['tests'].split(',')
        return json.dumps(fxt.run_multiple(listToRun))
    return None

def serve_task(req, sTask):
    reload(GLOBAL_STATE["modWork"])
    dictTask = get_task_dict()
    if sTask not in dictTask:
        raise ValueError("No such task: %s" % sTask)
    return json.dumps(dictTask[sTask].run())

def serve_updates(req, sUpdateTask):
    if sUpdateTask == "check":
        cmt = updatemanager.check_for_updates()
        return json.dumps(cmt and cmt.to_json())
    elif sUpdateTask == "install":
        return json.dumps({"success": updatemanager.deploy_updates()})
    return None

class TaskRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_safe_request(self, listHandlers):
        self.expand_aliases()
        for fxnHandler,regexp in listHandlers:
            try:
                match = regexp.match(self.path)
                if match is not None:
                    dictMatch = match.groupdict()
                    sOut = fxnHandler(self,**dictMatch)
                    if sOut is not None:
                        self.wfile.write(sOut)
                        return
            except Exception:
                import traceback
                self.send_error(500)
                sTb = traceback.format_exc()
                print sTb
                self.wfile.write("<pre>%s</pre>" % sTb)
                return
        self.send_error(404)

    def do_GET(self):
        self.do_safe_request([
            (serve_static,STATIC_RE),
            (serve_test,TEST_RE),
            (serve_metadata,METADATA_RE)])
    def do_POST(self):
        self.do_safe_request([(serve_test,TEST_RE), (serve_task,TASK_RE),
                              (serve_updates, UPDATES_RE)])
    def expand_aliases(self):
        if self.path in dictAlias:
            self.path = dictAlias[self.path]

def fork_httpd(httpd):
    tr = threading.Thread(target=httpd.serve_forever)
    tr.setDaemon(True)
    tr.daemon = True
    tr.start()
    return tr

def spawn(sDir, iPort, sStaticDirPath, sSelector, dblDelayMs=500):
    eventlog.init()
    dictConfig = loadconfig.load_config_file(sDir)
    GLOBAL_STATE["modTask"] = load_module(dictConfig["taskmodule"])
    GLOBAL_STATE["modTest"] = load_module(dictConfig["testmodule"])
    GLOBAL_STATE["modWork"] = load_module(dictConfig["workmodule"])
    GLOBAL_STATE["sStatic"] = sStaticDirPath
    GLOBAL_STATE["dictConfig"] = dictConfig
    httpd = BaseHTTPServer.HTTPServer(("localhost",iPort),TaskRequestHandler)
    tr = fork_httpd(httpd)
    sUrl = ("http://localhost:%d/" % iPort) + sSelector.lstrip("/")
    print sUrl
    webbrowser.open(sUrl)
    try:
        while True:
            tr.join(dblDelayMs)
    except KeyboardInterrupt:
        print "\nReceived SIGTERM, exiting..."
        sys.exit(-signal.SIGTERM)
        
def main(argv):
    parser = optparse.OptionParser()
    parser.add_option("-p", "--port", action="store", dest="port", type=int,
                      help="port to bind to.", default=DEFAULT_PORT)
    parser.add_option("-s", "--static-dir", action="store", dest="staticdir",
                      type=str, help="path to static content directory",
                      default=path.join(path.dirname(__file__),"./static/"))
    parser.add_option("-d", "--dir", action="store", dest="dir",
                      type=str, help="path to assignment directory",
                      default=path.abspath(CONFIG_DIR))
    parser.add_option("-l", "--selector", action="store", dest="selector",
                      type=str, help="HTTP URL selector for browser window",
                      default="")
    opts,args = parser.parse_args(argv)
    spawn(opts.dir, opts.port, opts.staticdir, opts.selector)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
