#!/bin/bash
# Is a heavy pipeline step running?  Prints its pid, or nothing.
#
# Deliberately NOT pgrep -f: that matches a substring of the whole command line, so any shell
# that merely *mentions* a script name -- an editor invocation, a progress check, the patch that
# hardened this very function -- counts as a competing process.  That silently blocked a
# supervisor gate twice.  Match structurally instead: argv[0] must be an interpreter and argv[1]
# must be one of our numbered pipeline scripts, which no wrapper shell satisfies.
# The first version only covered scripts 60-79 and so missed 52_core_matrix, whose
# 10 GB resident set then looked like unexplained memory pressure to the gate.
ps -eo pid=,args= | awk -v me=$$ '
  $1 != me && $2 ~ /(python|python3|python3\.8|python3\.9)$/ &&
  $3 ~ /scripts\/[0-9][0-9][a-z]?_[a-z0-9_]*\.py$/ { print $1; exit }
  $1 != me && $2 ~ /run_simulation\.py$/ { print $1; exit }'
