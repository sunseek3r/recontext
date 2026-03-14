import ast

class SemanticChunkExtractor(ast.NodeVisitor):
    def __init__(self, source_code, file_path):
        self.source_code = source_code
        self.file_path = file_path
        self.source_lines = source_code.splitlines()
        
        # Data structures to hold our parsed context
        self.imports = []
        self.classes = []
        self.functions = []

    def visit_Import(self, node):
        """Captures standard imports (e.g., import os)"""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Captures specific imports (e.g., from utils import auth)"""
        module = node.module if node.module else ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Extracts class definitions and their docstrings"""
        docstring = ast.get_docstring(node)
        
        self.classes.append({
            "file_path": self.file_path,
            "type": "class",
            "name": node.name,
            "docstring": docstring,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            # Grabbing the raw source code of the class
            "code": "\n".join(self.source_lines[node.lineno-1:node.end_lineno])
        })
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Extracts synchronous functions"""
        self._handle_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Extracts asynchronous functions"""
        self._handle_function(node)
        self.generic_visit(node)

    def _handle_function(self, node):
        """Helper to process both sync and async functions"""
        docstring = ast.get_docstring(node)
        
        # Extract just the signature to save precious context window tokens
        signature = self.source_lines[node.lineno - 1] 
        
        self.functions.append({
            "file_path": self.file_path,
            "type": "function",
            "name": node.name,
            "signature": signature,
            "docstring": docstring,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "code": "\n".join(self.source_lines[node.lineno-1:node.end_lineno])
        })


def parse_repository_file(file_content, file_path):
    """
    Main entry point to parse a single Python file into semantic chunks.
    """
    try:
        # ast.parse builds the syntax tree
        tree = ast.parse(file_content)
        
        # Initialize our visitor and walk the tree
        extractor = SemanticChunkExtractor(file_content, file_path)
        extractor.visit(tree)
        
        return {
            "status": "success",
            "imports": extractor.imports,
            "classes": extractor.classes,
            "functions": extractor.functions
        }
        
    except SyntaxError:
        # Built-in AST fails if the code has syntax errors. 
        # In a real IDE environment, developers are typing, so files might be incomplete.
        return {
            "status": "error",
            "message": f"SyntaxError in {file_path}. Consider fallback to Tree-sitter or regex."
        }