#!/usr/bin/env python3

import sys
import csv
import glob
import re
import coddLogParser

def getFilteredLogContent(log_filepath):
    with open(log_filepath, "r") as log_file:
        lines = log_file.readlines()

    toKeep = ("COMMAND", "SOLUTION", "COMPLETED", "TIMEOUT", "INFEASIBLE", "Maximum resident set size")
    filtered_lines = []
    for line in lines:
        if any(key in line for key in toKeep)  or line.strip() == "":
            filtered_lines.append(re.sub(r"\s+", " ", line).strip())

    # Join with newline characters
    return "\n".join(filtered_lines) + "\n"

log_filepath = sys.argv[1]
log_content = getFilteredLogContent(log_filepath)
#print(log_content)

# Parse log file
log_tree = coddLogParser.CoddOutputGrammar.parse(log_content)
log_data = coddLogParser.CoddOutputVisitor().visit(log_tree)
#print(log_data)

# Write statistics CSV
csv_filepath = log_filepath.replace(".txt",".csv")
csv_file = open(csv_filepath, "w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(log_data["header"])
csv_writer.writerows(log_data["rows"])
csv_file.close()
