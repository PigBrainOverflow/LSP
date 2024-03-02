import ast
from typing import Set, Dict, List
from .core import *
from .tf_analysis import *
from copy import deepcopy


#################################################
# utility functions
#################################################

def parse_file(filename: str) -> ast.Module:
    # filename must be an absolute path!
    # read
    with open(filename, "r", encoding="utf-8") as file:
        code = file.read()
    # parse
    return ast.parse(code, filename=filename)


# when parse the ast tree, we apply visitor pattern

#################################################
# visitors
#################################################

class TopVisitor(ast.NodeVisitor):
    _analyzer: Analyzer

    def __init__(self, analyzer: Analyzer):
        super().__init__()
        self._analyzer = analyzer

    def visit_Module(self, node: ast.Module):
        module_visitor = ModuleVisitor(self)
        module_visitor.visit(node)

    @property
    def analyzer(self) -> Analyzer:
        return self._analyzer

    @property
    def local_ids(self) -> Dict[str, Variable]:
        # the top visitor has an empty local scope
        return {}


class ModuleVisitor(ast.NodeVisitor):
    _father: ast.NodeVisitor
    _local_ids: Dict[str, Variable] # value is None if we don't care about it

    sensitive_nodetypes: Set[type] = {ast.Assign, ast.FunctionDef, ast.Return}

    def visit(self, node: ast.Module) -> Tensor | None:
        # visit all children sequentially
        # return the first return value
        for child in node.body:
            if isinstance(child, ast.Return):
                return self.visit_Return(child)
            if isinstance(child, ast.Assign):
                self.visit_Assign(child)
            elif isinstance(child, ast.FunctionDef):
                self.visit_FunctionDef(child)
            elif isinstance(child, ast.Return):
                self.visit_Return(child)

    def __init__(self, father: ast.NodeVisitor):
        super().__init__()
        self._father = father
        self._local_ids = deepcopy(father.local_ids)
        # we create a new local scope
        # independent from the father's

    # we only process assignment and function definition
    # omit other statements
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) > 1:
            self.analyzer.add_diag_msg("not support multi-assignment", lineno=node.lineno, col_offset=node.col_offset)
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            self.analyzer.add_diag_msg("not support tuple-assignment", lineno=node.lineno, col_offset=node.col_offset)
        val = node.value
        # process right-hand side
        retval_visitor = ExprVisitor(self)
        retval = retval_visitor.visit(val)
        if not isinstance(retval, Variable):    # we don't care about others
            retval = None
        self.local_ids[target.id] = retval # update variable / create a new one

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # it will add the new function to the local scope
        if node.decorator_list: # if decorated
            decval = None
            fst_decorator = node.decorator_list[0]  # we assume there's only one decorator
            if isinstance(fst_decorator, ast.Name): # e.g. @ewise_op_assert
                decorator_id = fst_decorator.id
                decval = self.analyzer.analyze_decorator_with_no_args(decorator_id)
            elif isinstance(fst_decorator, ast.Call):   # e.g. @conv2d_assert(...)
                decorator_id = fst_decorator.func.id
                keywords = fst_decorator.keywords
                decval = self.analyzer.analyze_decorator_with_args(decorator_id, keywords, lineno=node.lineno, col_offset=node.col_offset)
            self.local_ids[node.name] = decval
            if decval is not None and decval.name == "block_assert":
                # check the body of the function
                body_visitor = ModuleVisitor(self)
                # body_visitor has a new local scope
                # we must add the args to the local scope
                args = node.args
                for arg, input in zip(args.args, decval.kwargs["inputs"]):
                    body_visitor._local_ids[arg.arg] = Tensor(**input)
                retval = body_visitor.visit(node)   # this is the return value of the function
                self.analyzer.analyze_block_assert(retval, decval.kwargs["output"], lineno=node.lineno, col_offset=node.col_offset)

    def visit_Return(self, node: ast.Return) -> Tensor | None:
        retval_visitor = ExprVisitor(self)
        retval = retval_visitor.visit(node.value)
        # we only care about the tensor
        if isinstance(retval, Tensor):
            return retval
        return None

    @property
    def analyzer(self) -> Analyzer:
        return self._father.analyzer

    @property
    def local_ids(self) -> Dict[str, Variable]:
        return self._local_ids


class ExprVisitor(ast.NodeVisitor):
    # visit() returns
    # Variable: if we care about the variable
    # None: if we don't care about the variable
    # Other Types: constants, etc
    _father: ast.NodeVisitor

    def __init__(self, father: ast.NodeVisitor):
        super().__init__()
        self._father = father

    def visit_Call(self, node: ast.Call) -> Variable | None:
        func = node.func
        args = node.args

        # process function
        if isinstance(func, ast.Name):
            func_id = func.id
        else:   # not a name
            return None

        # process args
        retval_visitor = ExprVisitor(self)
        args_retvals = [
            retval_visitor.visit(arg)
            for arg in args
        ]

        return self.analyzer.analyze_call(self.local_ids, func_id, args_retvals, lineno=node.lineno, col_offset=node.col_offset)

    def visit_Name(self, node: ast.Name) -> Variable | None:
        # lookup id in local scope
        id = node.id
        if id in self.local_ids:
            return self.local_ids[id]
        return None

    def visit_Constant(self, node: ast.Constant) -> int | float:
        return node.value

    def visit_BinOp(self, node: ast.BinOp) -> Variable | None:
        op, left, right = node.op, node.left, node.right

        # process left and right
        retval_visitor = ExprVisitor(self)
        lretval = retval_visitor.visit(left)
        rretval = retval_visitor.visit(right)

        return self.analyzer.analyze_binop(op, lretval, rretval, lineno=node.lineno, col_offset=node.col_offset)

    @property
    def analyzer(self) -> Analyzer:
        return self._father.analyzer

    @property
    def local_ids(self) -> Dict[str, Variable]:
        return self._father.local_ids

