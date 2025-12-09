from collections import OrderedDict
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor


# COMMAND: /usr/bin/time -v timeout 600 ../cmake-build-release-ilyin/tsptw_mst_triangleRO1 ../data/tsptw/Solnon25_feasible/n51g60b50.005.txt 1024
# Command exited with non-zero status 124
#
# COMMAND: /usr/bin/time -v timeout 600 ../cmake-build-release-ilyin/tsptw_mst_triangleRO1 ../data/tsptw/Solnon25_feasible/n51g80b10.001.txt 1024
# P TIGHTEN: 0.351000 <P:1525.000000, D:459.000000, INC:(51)[49 23 33 32 12 1 27 22 42 44 7 28 29 4 31 40 15 13 5 45 43 36 9 14 30 35 38 2 19 20 50 47 26 8 24 17 39 37 6 21 3 18 10 34 48 41 46 16 25 11 0 ]> 1525.000000 1525.000000 + 0.000000 #18471
# P TIGHTEN: 0.002000 <P:835.000000, D:278.000000, INC:(21)[20 8 3 17 10 16 13 11 19 4 12 9 6 7 15 18 5 14 2 1 0 ]> 835.000000 835.000000 + 0.000000 #126
# Done(1024):1525.000000 #nodes:229/229 P/D:0/0 Time:0.351000/1.695000s LIM?:0 Seen:0

# COMMAND: /usr/bin/time -v timeout 600 ../cmake-build-release-ilyin/tsptw_mst_triangleRO1 ../data/tsptw/Solnon25_infeasible/n51g100b30.004.txt 1024
# Done(1024):2147483647.000000 #nodes:4496/4496 P/D:0/0 Time:0.000000/12.702000s LIM?:0 Seen:0



CoddBnBOutputGrammar = Grammar("""
    output = instace_output+
    instace_output = cmd_line solution_line* status_line mem_line nl*
    cmd_line = "COMMAND: /usr/bin/time -v timeout 600 ../cmake-build-release-ilyin/tsptw_" ("mst" / "greedy") "_triangleRO1 " instance " " int nl
    instance = "../data/tsptw/" word "/" filename
    solution_line   =  "P TIGHTEN: " best_time " <P:" best_cost rotl
    best_time = float
    best_cost = float
    status_line = (completed_line / timeout_line)
    completed_line  = "Done(" int "):" float " #nodes:" int "/" int " P/D:" int "/" int " Time:" float "/" float rotl
    timeout_line    = "Command exited with non-zero status" rotl
    mem_line = "Maximum resident set size (kbytes): " int
    line = str nl
    rotl = (str / " ")+ nl
    timestamp = "[" ws? float "s]"
    int_list = int ("," int)*
    str = ~"."+
    filename = (word / ".")+
    word = ~"[a-zA-Z0-9_]"+
    value = float / int
    int = ~"[-+]"? ~"[0-9]"+
    float = int "." int ("e" int)?
    nl = ~"\\n"
    ws = ~"[ \t]"+
    """)

class CoddBnBOutputVisitor(NodeVisitor):

    def __init__ (self):
        self.header =  ["Benchmark", "Instance", "Best Cost", "Best Time", "Search Time", "Nodes", "Timeout", "Memory"]
        self.row = None
        self.rows = []

    def visit_output(self, node, visited_children):
        return {"header": self.header,
                "rows" : self.rows}

    def visit_instace_output(self, node, visited_children):
        self.rows.append(self.row.values())
        self.row = None

    def visit_instance(self, node, visited_children):
        self.row = OrderedDict((key, None) for key in self.header)
        self.row["Benchmark"] = visited_children[1]
        self.row["Instance"]  = visited_children[3]

    def visit_nodes(self, node, visited_children):
        self.row["Nodes"]      = visited_children[0]

    def visit_solution_line(self, node, visited_children):
        self.row["Best Time"] = visited_children[1]
        self.row["Best Cost"] = visited_children[3]

    def visit_solve_time(self, node, visited_children):
        self.row["Search Time"] = visited_children[0]

    def visit_completed_line(self, node, visited_children):
        #"Done(" int "):" float " #nodes:" int "/" int " P/D:" int "/" int " Time:" float "/" float rotl
        #[None, 1024, None, 1334.0, None, 2666, None, 2666, None, 0, None, 0, None, 0.47, None, 2.714, None]
        self.row["Nodes"]      = visited_children[7]
        self.row["Search Time"] = visited_children[15]
        self.row["Timeout"]    = False # Timeout

    def visit_timeout_line(self, node, visited_children):
        self.row["Timeout"]       = True

    def visit_mem_line(self, node, visited_children):
        #Maximum resident set size (kbytes): 28267720
        self.row["Memory"] = visited_children[1]

    def visit_timestamp(self, node, visited_children):
        return visited_children[2]

    def visit_int_list(self, node, visited_children):
        return node.text.replace(",", "")

    def visit_value(self, node, visited_children):
        return visited_children[0]

    def visit_float(self, node, visited_children):
        return float(node.text)

    def visit_int(self, node, visited_children):
        return int(node.text)

    def visit_word(self, node, visited_children):
        return node.text

    def visit_str(self, node, visited_children):
        return node.text

    def visit_rotl(self, node, visited_children):
        return None
    
    def generic_visit(self, node, visited_children):
        if len(visited_children) == 0:
            return None
        else:
            return visited_children[0]
