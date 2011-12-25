#!/usr/bin/env python

"""
eventlog.py -- create records of actions including file modification,
unit test success/failure, and running of tasks.
"""

import datetime
import os
from os import path
import sqlite3
import time

import loadconfig

FILENAME = "events.sqlite"

SCHEMA_TEMPLATE = (
("test_invocation", """
CREATE TABLE test_invocation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp NUMERIC NOT NULL,
  test_name TEXT NOT NULL,
  test_id INTEGER NOT NULL,
  success INTEGER NOT NULL,
  traceback TEXT)"""),
("task_invocation", """
CREATE TABLE task_invocation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp NUMERIC NOT NULL,
  task_name TEXT NOT NULL,
  task_id INTEGER NOT NULL,
  success INTEGER NOT NULL,
  traceback TEXT)"""),
)

CONFIG_DIR = loadconfig.get_config_dir()

def get_time_now():
    return time.mktime(datetime.datetime.now().timetuple())
    
def open_conn():
    kwargs = {"isolation_level": None}
    conn = sqlite3.connect(path.join(CONFIG_DIR, FILENAME), **kwargs)
    return conn

def list_tables(conn):
    c = conn.execute("SELECT name FROM sqlite_master")
    return [row[0] for row in c if not row[0].startswith("sqlite_")]

def build_db(listSTmpl, conn):
    setSTableName = set(list_tables(conn))
    for sTableName,sTableTmpl in listSTmpl:
        if sTableName not in setSTableName:
            sTableSchema = sTableTmpl % {}
            conn.execute(sTableSchema)

def init():
    with open_conn() as conn:
        build_db(SCHEMA_TEMPLATE, conn)

def add_invocation(conn, sTableName, sNameField, dictProperties):
    listRequiredFields = ["success"] + [sTmpl % sNameField
                                        for sTmpl in "%s_name", "%s_id"]
    for sField in listRequiredFields:
        if sField not in dictProperties:
            raise TypeError("Missing invocation property: %s" % sField)

    listFieldNames = ["timestamp",
                      "%s_name" % sNameField,
                      "%s_id" % sNameField,
                      "success",
                      "traceback"]
    listFieldValues = ([get_time_now()] +
                       [dictProperties.get(sFieldName)
                        for sFieldName in listFieldNames[1:]])
    sFmtTmpl = """INSERT INTO %(sTableName)s
    (%(sFieldNames)s)
    VALUES (?,?,?,?,?)"""
    sSql = sFmtTmpl % {"sTableName": sTableName,
                       "sNameField": sNameField,
                       "sFieldNames": ", ".join(listFieldNames)}
    c = conn.cursor()
    c.execute(sSql, listFieldValues)

def add_test_invocation(conn, dictProperties):
    return add_invocation(conn, "test_invocation", "test", dictProperties)

def add_task_invocation(conn, dictProperties):
    return add_invocation(conn, "task_invocation", "task", dictProperties)

def test_success(sTestName, ixTestId):
    dictProperties = {"success": True, "traceback": None,
                      "test_name": sTestName, "test_id": int(ixTestId)}
    with open_conn() as conn:
        return add_test_invocation(conn, dictProperties)

def test_failure(sTestName, ixTestId, sTraceback):
    dictProperties = {"success": False, "traceback": sTraceback,
                      "test_name": sTestName, "test_id": int(ixTestId)}
    with open_conn() as conn:
        return add_test_invocation(conn, dictProperties)

def task_success(sTaskName, ixTaskId):
    dictProperties = {"success": True, "traceback": None,
                      "task_name": sTaskName, "task_id": int(ixTaskId)}
    with open_conn() as conn:
        return add_task_invocation(conn, dictProperties)

def task_failure(sTaskName, ixTaskId, sTraceback):
    dictProperties = {"success": False, "traceback": sTraceback,
                      "task_name": sTaskName, "task_id": int(ixTaskId)}
    with open_conn() as conn:
        return add_task_invocation(conn,dictProperties)

def list_tests(conn):
    c = conn.execute("SELECT * FROM test_invocation")
    return [r for r in c]

def list_tasks(conn):
    c = conn.execute("SELECT * FROM task_invocation")
    return [r for r in c]

def fmt_invocation(tplInv):
  listLines = zip(("ID", "Run At", "Name", None, "Success", "Traceback",),
                  tplInv)
  for sName,sValue in listLines:
      if sName is not None:
          print "%s\t%s" % (sName, sValue)
          
def main(argv):
    conn = open_conn()
    #build_db(SCHEMA_TEMPLATE, conn)
    #test_success("passing_test", 1)
    #test_failure("failing_test", 2, "tb")
    #task_success("passing_task", 1)
    #task_failure("failing_task", 2, "tb")
    print "Tests:"
    for tplInv in list_tests(conn):
        fmt_invocation(tplInv)
        print ""
    print "Tasks:"
    for tplInv in list_tasks(conn):
        fmt_invocation(tplInv)
        print ""
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
