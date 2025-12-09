#!/usr/bin/env python3

import sys
import csv
import glob
import re
import routeOptLogParser


def getFilteredLogContent(log_filepath):
    with open(log_filepath, "r") as log_file:
        lines = log_file.readlines()

    ansi_escape = re.compile(r'\x1B\[[0-9;?]*[A-Za-z]')
    toKeep = (
        "COMMAND",
        "<UB",
        "<LB",
        "<Elapsed",
        "<Nodes",
        "non-zero status",
        "terminated by",
        "Maximum resident set size")
    filtered_lines = []
    for line in lines:
        line = ansi_escape.sub('', line)
        line = line.replace("1e+06", "1000000")
        line = line.replace("1e+09", "1000000000")
        if any(key in line for key in toKeep):
            line = ansi_escape.sub('', line)
            line = re.sub(r"\s+", " ", line).strip()
            if 'Command exited' in line and not line.startswith('Command exited'):
                line = line[line.index('Command exited'):]
            if 'Command terminated' in line and not line.startswith('Command terminated'):
                line = line[line.index('Command terminated'):]
            if line != "":
                filtered_lines.append(line)

    # Join with newline characters
    return "\n".join(filtered_lines) + "\n"

log_filepath = sys.argv[1]
log_content = getFilteredLogContent(log_filepath)
#print(log_content)

# Parse log file
log_tree = routeOptLogParser.RouteOptOutputGrammar.parse(log_content)
log_data = routeOptLogParser.RouteOptOutputVisitor().visit(log_tree)
#print(log_data)

# Write statistics CSV
csv_filepath = log_filepath.replace(".txt",".csv")
csv_file = open(csv_filepath, "w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(log_data["header"])
csv_writer.writerows(log_data["rows"])
csv_file.close()
