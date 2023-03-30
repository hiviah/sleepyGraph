#!/usr/bin/env python

import sys
import re

import xdot
import gtk
import gtk.gdk

from zipfile import ZipFile, BadZipfile
from optparse import OptionParser
from collections import deque

from pygraphviz import AGraph

def fatal(msg, errcode=1):
    print >> sys.stderr, msg
    exit(errcode)

def gtkMessageBox(message, type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK, 
                  title='VerySleepy Callgraph Viewer'):
        dlg = gtk.MessageDialog(type=type,
                                message_format=message,
                                buttons=buttons)
        dlg.set_title(title)
        dlg.run()
        dlg.destroy()

def gtkMessageError(message, buttons=gtk.BUTTONS_OK,
                  title='VerySleepy Callgraph Viewer'):
    gtkMessageBox(message, gtk.MESSAGE_ERROR, buttons, title)
    
class GraphWindow(xdot.DotWindow):

    def __init__(self):
        xdot.DotWindow.__init__(self)
        self.widget.connect('clicked', self.on_url_clicked)
        self.db = None
        self.minPercent = 1
        self.maxDepth = 4
        
        sw = gtk.ScrolledWindow()
        #model is: key, func, inclusive, exclusive
        self.store = gtk.ListStore(str, str, int, int)
        
        self.symbolView = gtk.TreeView(self.store)
        self.symbolView.connect("row-activated", self.symbol_activated)
        self.symbolView.set_rules_hint(True)
        rendererText = gtk.CellRendererText()
        
        column = gtk.TreeViewColumn("Symbol", rendererText, text=1)
        column.set_sort_column_id(1)    
        self.symbolView.append_column(column)
        
        column = gtk.TreeViewColumn("Inclusive", rendererText, text=2)
        column.set_sort_column_id(2)
        self.symbolView.append_column(column)
        
        column = gtk.TreeViewColumn("Exclusive", rendererText, text=3)
        column.set_sort_column_id(3)
        self.symbolView.append_column(column)
        
        sw.add(self.symbolView)
        self.vpaned.add2(sw)
        self.symbolView.show()
        self.vpaned.set_property("position", 256)
        self.show_all()


    def open_db_file(self, filename):
        self.db = Database(filename)
        self.callgraph = Callgraph(self.db.symTab, self.db.stacks)
        for symbol in self.db.symTab.values():
            self.store.append([symbol.key, symbol.func, symbol.inclusive, symbol.exclusive])
    
    def open_file(self, filename):
        try:
            self.open_db_file(filename)
            self.make_graph()
        except (IOError, BadZipfile), ex:
            gtkMessageError(str(ex))

    def on_open(self, action):
        chooser = gtk.FileChooserDialog(title="Open VerySleepy File",
                                        action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_OPEN,
                                                 gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        filter = gtk.FileFilter()
        filter.set_name("Sleepy dot files")
        filter.add_pattern("*.sleepy")
        chooser.add_filter(filter)
        filter = gtk.FileFilter()
        filter.set_name("All files")
        filter.add_pattern("*")
        chooser.add_filter(filter)
        if chooser.run() == gtk.RESPONSE_OK:
            filename = chooser.get_filename()
            chooser.destroy()
            self.open_file(filename)
        else:
            chooser.destroy()
    
    def on_url_clicked(self, widget, url, event):
        self.make_graph(self.db.symTab[url])
        return True
    
    def make_graph(self, node=None):
        self.set_dotcode(self.callgraph.toDotFormat(node, maxDepth=self.maxDepth,
                          minPercent=self.minPercent))
    
    def symbol_activated(self, widget, row, col):
        self.make_graph(self.db.symTab[self.store[row][0]])

class Symbol:
    """ Keeps info about a single symbol - function/method """
    
    """ Regexp to parse symbol line """
    symRe = re.compile(r'^(\S+)\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+(\d+)')
    
    def __init__(self, symLine):
        m = Symbol.symRe.match(symLine.strip())
        parts = [part.decode("UTF-8", "ignore") for part in m.groups()]
        
        (self.key, self.dll, self.func, self.srcFile, self.srcLine) = parts
        self.inclusive = 0
        self.exclusive = 0
        
    @staticmethod
    def prettyFunc(name, minSplit=30):
        """ Splits function name into a multi-line string with literal \n """
        res = ""
        lastPos = 0
        for m in re.finditer("\w+", name):
            if m.start() - lastPos >= minSplit:
                res += name[lastPos:m.start()] + r"\n"
                lastPos = m.start()
        res += name[lastPos:]
        return res
    
    def toShortStr(self, comment=""):
        return "%s %s\\n%s\\nexcl %d, incl %d" % \
            (comment, "", Symbol.prettyFunc(self.func), self.exclusive,
             self.inclusive)
    
    def __str__(self):
        return "<%s: function '%s', excl %d, incl %d>" % \
            (self.key, self.func, self.exclusive, self.inclusive)
    
    def __repr__(self):
        return str(self)

class WeightedSymbol(Symbol):
    def __init__(self, sym, relativeCost):
        (self.key, self.dll, self.func, self.srcFile, self.srcLine) = \
        (sym.key,  sym.dll,  sym.func,  sym.srcFile,  sym.srcLine)
        self.inclusive = relativeCost
        self.exclusive = relativeCost
    
class SymbolNode(Symbol):
    def __init__(self, sym):
        (self.key, self.dll, self.func, self.srcFile, self.srcLine) = \
        (sym.key,  sym.dll,  sym.func,  sym.srcFile,  sym.srcLine)
        
        #children and cyclChildren are disjunct sets (same for parents and
        #cyclParents)
        self.children = set()
        self.parents = set()
        self.cyclChildren = set()
        self.cyclParents = set()
        
        self.inclusive = 0
        self.exclusive = 0

class Callgraph:
    def __init__(self, symTab, stacks):
        self.symToNode = {} #maps Symbol from Database to SymbolNode
        rootSymbol = stacks[0].path[-1]
        self.root = SymbolNode(rootSymbol)
        self.symToNode[rootSymbol] = self.root
        self.totalCost = 0
        
        for stack in stacks:
            path = stack.path
            seenSyms = set() #seen Symbols to detect cycles
            lastSym = None #last Symbol to skip immediate recursive calls
            parentSymNode = None #parent of currently processed node in graph
            
            for symbol in reversed(path):
                if symbol == lastSym:
                    continue #recursive call
                
                symbolNode = self.symToNode.setdefault(symbol, SymbolNode(symbol))
                if symbol in seenSyms: #cycle, but not direct recursive call
                    if parentSymNode:
                        parentSymNode.cyclChildren.add(SymbolNode)
                        symbolNode.cyclParents.add(parentSymNode)
                else:
                    if parentSymNode:
                        parentSymNode.children.add(symbolNode)
                        symbolNode.parents.add(parentSymNode)
                    symbolNode.inclusive += stack.cost
                
                seenSyms.add(symbol)
                lastSym = symbol
                parentSymNode = symbolNode
            
            #we'll add exclusive cost only to deepest symbol
            deepestNode = self.symToNode[path[0]]
            deepestNode.exclusive += stack.cost
            self.totalCost += stack.cost
    
    def dfs(self, start=None, maxDepth=sys.maxint, pred=lambda node: True):
        """ Returns generator that iterates over graph in DFS order.
        start is Symbol or SymbolNode to start from, None means start from root.
        maxDepth is maximum depth to display, start node is considered to be at
        depth 0 (i.e. maxDepth == 0 will return only start node).
        
        Predicate pred returns True if given node should be accepted. If node
        is not accepted, neither are its children.
        
        Returns tuples (SymbolNode current, int depth, SymbolNode parent).
        First SymbolNode is the actual position in graph, second is depth and
        the second SymbolNode is the parent through which we got to this
        current node.
        """
        if start is None:
            start = self.root
        else:
            start = self.symToNode.get(start, start) #turn start into SymbolNode
        
        #We can have cycle even if we detected cycles in separate stack traces
        #since two callstacks can create a "false" cycle, e.g.
        #A->B->C and A->C->B will result in B<->C cycle in final graph
        visited = set() #set of visited SymbolNodes
        #deque of (SymbolNode current, depth, SymbolNode parent)
        toVisit = deque([(start,0, None)])
        
        descendPred = lambda node: pred(node) and (node not in visited)
        
        while len(toVisit) > 0:
            (node, nodeDepth, parentNode) = toVisit.pop()
            visited.add(node)
            yield (node, nodeDepth, parentNode)
            
            if nodeDepth < maxDepth:
                children = [child for child in node.children if descendPred(child)]
                childCount = len(children)
                toVisit.extend(zip(children, [nodeDepth+1] * childCount, [node] * childCount))
    
    def toDotFormat(self, start=None, maxDepth=sys.maxint, minPercent=1,
                    withParents=True):
        """ Returns the graph starting from node start in Graphviz Dot format.
        If start is None, starts at root. maxDepth specifies maximum depth as
        in dfs() method.
        """
        dotGraph = AGraph(directed=True, strict=True)
        visited = set([start])
        threshold = float(minPercent)/100.0*self.totalCost
        first = True
        
        thresPred = lambda node: node.inclusive >= threshold
        
        dotGraph.node_attr.update(shape="rectangle", style="filled")
        for (node, depth, parent) in self.dfs(start, maxDepth, thresPred):
            if first:
                dotGraph.add_node(node.key, color="blue", **self._nodeParams(node))
                first = False
            else:
                dotGraph.add_node(node.key, **self._nodeParams(node))
            
            if parent is not None: #parent == None iff node == start
                dotGraph.add_edge(parent.key, node.key)
            if withParents:
                for nodeParent in (n for n in node.parents if thresPred(n) and n.key != node.key):
                    dotGraph.add_node(nodeParent.key, **self._nodeParams(nodeParent))
                    dotGraph.add_edge(nodeParent.key, node.key)
        
        return dotGraph.to_string()
        
    def _nodeParams(self, node):
        """ Returns dict of node atrributes to be passed to graphviz """
        percentage = 100.0 * node.inclusive / self.totalCost
        percentStr = "%.1f %% incl" % percentage
        return {
                "label": str(node.toShortStr(percentStr)),
                "URL": str(node.key),
                "fillcolor": "0.529 0.19 %.3f" % (1-(percentage/200.0))
                }

class Callstack:
    def __init__(self, callLine, symTab):
        parts = callLine.split(" ")
        self.cost = int(parts[0])
        #convert textual path to tuple of Symbols
        self.path = tuple([symTab[symName] for symName in parts[1:]])
        self.__hashVal = hash(self.cost) ^ hash(self.path)
        
    def __hash__(self):
        return self.__hashVal
    
    def __eq__(self, other):
        return self.cost == other.cost and self.path == other.path
    
    def __str__(self):
        return "%s" % [sym.key for sym in reversed(self.path)]
    
    def __repr__(self):
        return str(self)

class Database:
    
    def __init__(self, zippedFname):
        self.symTab = {}
        self.nodes = {}
        self.symToStacks = {} #maps Symbol to set of Callstacks it is present in
        self.totalCost = 0
        self.stacks = [] #list of Callstacks
        
        #Maps tuple of suffix (..., sym1, sym0) to set of Callstacks with
        #stack (..., symN, sym1, sym0) it is suffix of (i.e. every member of
        #the value set was called through stack that is the key of the map).
        #The tuple members are strings "symXXX" as present in the callstack
        self.suffixStackMap = {}
        
        try:
            zipped = ZipFile(zippedFname)
            #ZipFile.open() would be nicer, but needs python 2.6
            syms = zipped.read("symbols.txt").splitlines()
            calls = zipped.read("callstacks.txt").splitlines()
            zipped.close()
        except (BadZipfile, KeyError):
            raise IOError, \
                "Input file %s is corrupt or does not contain needed files" % \
                  zippedFname
            
        self._parseSymTab(syms)
        self._parseCallstacks(calls)
    
    def callers(self, sym):
        """ Returns set of all callers of given Symbol. """
        result = set()
        stacks = self.symToStacks[sym]
        return result
    
    def callees(self, sym):
        """ Returs set of all immediate callees of given Symbol or
        WeightedSymbol.
        """
        result = set()
        counts = {}
        stacks = self.symToStacks[sym]
        
        for stack in stacks:
            path = list(stack.path)
            idx = path.index(sym)
            if idx > 0:
                previousSym = path[idx-1]
                counts[previousSym] = counts.get(previousSym, 0) + stack.cost
        
        for sym, cost in counts.iteritems():
            result.add(WeightedSymbol(sym, cost))
        return result

    def _parseSymTab(self, symLines):
        """ Converts all lines of symbol file to symbol mappings """
        for line in symLines:
            sym = Symbol(line)
            self.symTab[sym.key] = sym

    def _parseCallstacks(self, calls):
        """ Parses callstack lines and computes symbol costs.
        Creates a map mapping each symbol to set of callstacks it is present in.
        """
        for line in calls:
            stack = Callstack(line, self.symTab)
            self.stacks.append(stack)
            self.totalCost += stack.cost
            
            #call traces are listed from deepest to main()
            stack.path[0].exclusive += stack.cost
            seenSyms = set() #prevent wrong inclusive counts caused by recursion
            
            stackOrder = \
                (sym for sym in reversed(stack.path) if sym not in seenSyms)
            stackSuffix = []
            for methodSymbol in stackOrder:
                seenSyms.add(methodSymbol)
                methodSymbol.inclusive += stack.cost

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--min-ratio",
        action="store", dest="minRatio", default=0.5)

    (opts, args) = parser.parse_args(sys.argv)
    try:
        opts.minRatio = float(opts.minRatio)
        assert(0 <= opts.minRatio <= 100)
    except (ValueError, AssertionError):
        fatal("min ratio is not a float number 0 <= x <= 100")
    
    window = GraphWindow()
    if len(sys.argv) > 1:
        window.open_file(sys.argv[1])
    window.connect('destroy', gtk.main_quit)
    gtk.main()
    
    #if len(args) < 2:
    #    fatal("No file argument given")
    #    
    #db = Database(args[1])
    #graph = Callgraph(db.symTab, db.stacks)
    #
    #dotStr = graph.toDotFormat(None, 15)
    #print dotStr
    #differentNodes = set()
    #for (node, depth) in graph.dfs(None, 10):
    #    print "%s%s" % (" " * depth*2, node)
    #    differentNodes.add(node)
    #print "Distinct node count: %d" % len(differentNodes)
    #print "Total node count in graph: %d" % len(graph.symToNode)
    #suff = [x for x in db.stacks if [s.key for s in x.path[-7:]] == ["sym1119", "sym1112", "sym1104", "sym1",  "sym2",  "sym1404",  "sym0"]]
    #for st in suff:
    #    print "%s" % st
    #print "%s" % repr(graph.root)
    #for ch in graph.root.children:
    #    print " -- %s" % repr(ch)
    #print "total %d" % db.totalCost
    #
    #print "Inclusive"
    #nodes = sorted(db.symTab.values(), key=lambda n: n.inclusive, reverse=True)
    #topFuncs = [db.symTab[n.key] for n in nodes[0:20]]
    #for f in topFuncs:
    #    print "%d %d %s" % (f.inclusive, f.exclusive, f.func)
    #
    #print "\n\n===Exclusive==="
    #nodes = sorted(db.symTab.values(), key=lambda n: n.exclusive, reverse=True)
    #topFuncs = [db.symTab[n.key] for n in nodes[0:50]]
    #for f in topFuncs:
    #    print "%d %d %s" % (f.inclusive, f.exclusive, f.func)
    #
    #print repr(db.symTab["sym1036"])
    #for s in db.callees(db.symTab["sym1036"]): print repr(s)