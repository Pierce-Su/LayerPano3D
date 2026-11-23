"""CPU regression: actual native glog lifetime across sample-local constructors.

Extract only the constructors to avoid unrelated GPU/model/image imports. Uses
installed instaOmniDepth, not a mock. A duplicate native shutdown aborts Python.
"""
import ast
import atexit
from pathlib import Path
from instaOmniDepth import depthmapAlign

repo = Path(__file__).resolve().parents[1]
for name, args in [
    ('utils/depth_alignment.py', (None, '', [])),
    ('submodules/360monodepth/code/python/src/utility/depthmap_align.py', ()),
]:
    tree = ast.parse((repo / name).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DepthmapAlign')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    cls.body = [init]
    env = {'depthmapAlign': depthmapAlign}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), name, 'exec'), env)
    env['DepthmapAlign'](*args)
    env['DepthmapAlign'](*args)
# Exercise real callbacks now. Native shutdown must execute only once.
atexit._run_exitfuncs()
print('PASS: four actual sample-local constructors, one native shutdown, normal exit', flush=True)
