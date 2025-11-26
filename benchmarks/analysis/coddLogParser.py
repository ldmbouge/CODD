from collections import OrderedDict
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

CoddOutputGrammar = Grammar("""
    output = instace_output+
    instace_output = header solution_line* status_line stat_line+ nl?
    header = cmd_line instance_line line line
    cmd_line = "COMMAND: " line
    instance_line = "Instance: ../data/tsptw/" word "/" str nl
    solution_line = timestamp " SOLUTION | Visited = " int " | Cost = " float " | Value = " int_list nl
    status_line = (completed_line / timeout_line / infeasable_line)
    stat_line = line
    completed_line = timestamp " COMPLETED | Visited = " int " | Cost = " float " | Value = " int_list nl
    timeout_line = timestamp " TIMEOUT | Visited = " int nl
    infeasable_line = timestamp " INFEASIBLE | Visited = " int nl
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

class CoddOutputVisitor(NodeVisitor):

    def __init__ (self):
        self.header =  ["Benchmark", "Instance", "Best Cost", "Best Time", "Proof Time", "Nodes", "Timeout"]
        self.current_row = None
        self.rows = []

    def visit_output(self, node, visited_children):
        return {"header": self.header,
                "rows" : self.rows}

    def visit_instace_output(self, node, visited_children):
        self.rows.append(self.row.values())
        self.row = None

    def visit_instance_line(self, node, visited_children):
        # Instance: ../data/tsptw/Solnon25_feasible/n21g100b10.003.txt
        self.row = OrderedDict((key, None) for key in self.header)
        self.row["Benchmark"] = visited_children[1]
        self.row["Instance"]  = visited_children[3]

    def visit_solution_line(self, node, visited_children):
        # [   0.01s] SOLUTION     | Visited =         51 | Cost =  753.00 | Value = 18,2,20,...
        self.row["Best Time"] = visited_children[0]
        self.row["Best Cost"] = visited_children[4]

    def visit_completed_line(self, node, visited_children):
        #[   0.01s] COMPLETED    | Visited =         72 | Cost =  755.00 | Value = 2,7,16,15,13,18,9,19,20,14,10,5,6,17,3,12,8,11,4,1,0
        self.row["Proof Time"] = visited_children[0]
        self.row["Nodes"]      = visited_children[2]
        self.row["Timeout"]    = False # Timeout

    def visit_timeout_line(self, node, visited_children):
        #[ 600.48s] TIMEOUT      | Visited = 9468208671
        self.row["Nodes"] = visited_children[2]
        self.row["Timeout"]       = True

    def visit_infeasable_line(self, node, visited_children):
        #[   0.00s] INFEASIBLE   | Visited =          6
        self.row["Proof Time"] = visited_children[0]
        self.row["Nodes"]      = visited_children[1]
        self.row["Timeout"]    = False # Timeout

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
    
    def generic_visit(self, node, visited_children):
        if len(visited_children) == 0:
            return None
        else:
            return visited_children[0]
