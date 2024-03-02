from . import diagnose
import sys

#################################################
# ENTRY
#################################################
if __name__ == "__main__":
    # test_file = sys.argv[1]
    test_file = "C:\\Users\\xiaokong\\Desktop\\lsp-2.0\\tests\\resnet.py"
    tree = diagnose.parse_file(test_file)
    analyzer = diagnose.Analyzer()
    top_visitor = diagnose.TopVisitor(analyzer)
    top_visitor.visit(tree)
    print("####################")
    print("diagnostic messages: ")
    print(analyzer.diag_msgs)