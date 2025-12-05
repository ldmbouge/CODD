from collections import OrderedDict
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

DidpOutputGrammar = Grammar("""
    output = instace_output+
    instace_output = cmd_line solution_line* status_line+ mem_line nl*
    cmd_line = "COMMAND: /usr/bin/time -v /home/fabio/didp-rust-models/target/release/tsptw_rpid -t 600 " instance nl
    instance = "../data/tsptw/" word "/" str
    solution_line = "New primal bound: " int ", expanded: " int ", generated: " int ", elapsed time: " float "s." nl
    status_line = completed_line / timeout_line / infeasable_line / search_time_line / expanded_line
    completed_line = "optimal cost: " int nl
    timeout_line = "Time limit reached." nl
    search_time_line = "Search time: " float "s" nl
    infeasable_line = "The problem is infeasible." nl
    expanded_line = "Expanded: " int nl
    mem_line = "Maximum resident set size (kbytes): " int
    line = str nl
    timestamp = "[" ws? float "s]"
    int_list = int ("," int)*
    str = ~"."+
    word = ~"[a-zA-Z0-9_]"+
    value = float / int
    int = ~"[-+]"? ~"[0-9]"+
    float = int "." int ("e" int)?
    nl = ~"\\n"
    ws = ~"[ \t]"+
    """)

class DidpOutputVisitor(NodeVisitor):

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
        #../data/tsptw/Solnon25_feasible/n21g100b10.003.txt
        self.row = OrderedDict((key, None) for key in self.header)
        self.row["Benchmark"] = visited_children[1]
        self.row["Instance"]  = visited_children[3]

    def visit_solution_line(self, node, visited_children):
        # New primal bound: 720, expanded: 996825, generated: 2176106, elapsed time: 1.529058793s.
        self.row["Best Time"] = visited_children[7]
        self.row["Best Cost"] = visited_children[1]

    def visit_completed_line(self, node, visited_children):
        #optimal cost: 1023
        self.row["Timeout"] = False

    def visit_timeout_line(self, node, visited_children):
        #Time limit reached.
        self.row["Timeout"] = True

    def visit_infeasable_line(self, node, visited_children):
        #The problem is infeasible.
        self.row["Timeout"] = False

    def visit_search_time_line(self, node, visited_children):
        #Search time: 603.090286512s
        self.row["Search Time"] = visited_children[1]

    def visit_expanded_line(self, node, visited_children):
        self.row["Nodes"] = visited_children[1]

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
