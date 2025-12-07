from collections import OrderedDict
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

# COMMAND: /usr/bin/time -v /home/fabio/RouteOpt/packages/application/cvrp/bin/benchmarks_tw_single ../data/tsptw/Solnon25_feasible/n21g100b10.001.txt
# <Solution>
# <UB= 835>
# <LB= 835>
# <Elapsed Time= 0.8>
# <Nodes Explored= 1>
# <Global Gap= 0%>


RouteOptOutputGrammar = Grammar("""
    output = instace_output+
    instace_output = cmd_line solution_line* mem_line nl*
    cmd_line = "COMMAND: /usr/bin/time -v " "timeout 600 "? "/home/fabio/RouteOpt/packages/application/cvrp/bin/benchmarks_tw_single " instance nl
    instance = "../data/tsptw/" word "/" str
    ub_line = "<UB=" ws? int ">" nl
    lb_line = "<LB=" ws? float ">" nl
    search_time_line = "<Elapsed Time=" ws? float ">" nl
    expanded_line = "<Nodes Explored=" ws? int ">" nl
    solution_line = ub_line / lb_line / search_time_line / expanded_line
    mem_line = "Maximum resident set size (kbytes): " int nl
    line = str nl
    str = ~"."+
    word = ~"[a-zA-Z0-9_]"+
    value = float / int
    int = ~"[-+]"? ~"[0-9]"+
    float = (int "." int ("e" int)?) / int
    nl = ~"\\n"
    ws = ~"[ \t]"+
    """)

class RouteOptOutputVisitor(NodeVisitor):

    def __init__ (self):
        self.header =  ["Benchmark", "Instance", "Best Cost", "Best Time", "Search Time", "Nodes", "Timeout", "Memory"]
        self.row = None
        self.rows = []
        self.ub = None
        self.lb = None

    def visit_output(self, node, visited_children):
        return {"header": self.header,
                "rows" : self.rows}

    def visit_instace_output(self, node, visited_children):
        self.rows.append(self.row.values())
        self.row = None
        self.up = None

    def visit_instance(self, node, visited_children):
        #../data/tsptw/Solnon25_feasible/n21g100b10.003.txt
        self.row = OrderedDict((key, None) for key in self.header)
        self.row["Benchmark"] = visited_children[1]
        self.row["Instance"]  = visited_children[3]
        self.ub = None
        self.row["Best Cost"] = None

    def visit_ub_line(self, node, visited_children):
        self.ub = visited_children[2]

    def visit_lb_line(self, node, visited_children):
        self.lb = visited_children[2]
        if self.lb == self.ub:
            self.row["Best Cost"] = self.lb
            self.row["Timeout"] = False
        else
            self.row["Timeout"] = True

    def visit_search_time_line(self, node, visited_children):
        #Search time: 603.090286512s
        self.row["Search Time"] = visited_children[2]

    def visit_expanded_line(self, node, visited_children):
        #optimal cost: 1023
        self.row["Nodes"] = visited_children[2]

    def visit_mem_line(self, node, visited_children):
        #Maximum resident set size (kbytes): 28267720
        self.row["Memory"] = visited_children[1]

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
    
    def generic_visit(self, node, visited_children):
        if len(visited_children) == 0:
            return None
        else:
            return visited_children[0]
