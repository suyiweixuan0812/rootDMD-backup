"""Remove HPSv2's unused turtle import in memory; never edit shared packages."""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import sys

TARGET = 'hpsv2.src.open_clip.factory'

class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path)
        tree = ast.parse(source, filename=self.path)
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'turtle']
        assert len(imports) == 1 and [(a.name, a.asname) for a in imports[0].names] == [('forward', None)]
        assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == 'forward' for n in ast.walk(tree)), 'Unused-import repair no longer applicable'
        tree.body.remove(imports[0])
        print('HPS_HEADLESS_UNUSED_IMPORT_REMOVED sha256=' + hashlib.sha256(source).hexdigest(), flush=True)
        return compile(tree, self.path, 'exec')

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.origin.endswith('.py')
        spec.loader = _Loader(fullname, spec.origin)
        return spec

def prepare_hps():
    if TARGET in sys.modules:
        return
    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        __import__(TARGET)
    finally:
        sys.meta_path.remove(finder)
