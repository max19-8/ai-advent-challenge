"""Проверяем dedupe на целых числах в ограниченном подмножестве Python."""
import ast
import json
import subprocess
import sys
from pathlib import Path


CASES = [([], []), ([3, 1, 3, 2, 1], [3, 1, 2]), ([5, 5, 5], [5]),
         ([-1, 0, -1, 2], [-1, 0, 2]), ([9, 3, 1], [9, 3, 1])]
RULES = (
    'Реализуй одну функцию def dedupe(values): для списка целых чисел. '
    'Допустимы локальные переменные, for, if, in/not in, return, списки, '
    'set(), list(), dict.fromkeys(), append() и add(). Без импортов, аннотаций, '
    'декораторов, вспомогательных функций или вызовов за пределами функции.'
)
ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.For, ast.If,
           ast.Assign, ast.Name, ast.Load, ast.Store, ast.Compare, ast.In, ast.NotIn,
           ast.Return, ast.List, ast.Call, ast.Attribute, ast.Expr, ast.Constant,
           ast.Continue, ast.Pass)
BUILTINS = {'set': set, 'list': list, 'dict': dict}


def check_source(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Ожидалась одна функция dedupe(values).')
    fn = tree.body[0]
    if (fn.name != 'dedupe' or fn.decorator_list or fn.returns or fn.args.defaults
            or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs
            or [a.arg for a in fn.args.args] != ['values'] or fn.args.args[0].annotation):
        raise ValueError('Требуется сигнатура def dedupe(values):')
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED) or isinstance(node, ast.FunctionDef) and node is not fn:
            raise ValueError(f'Конструкция вне учебного подмножества: {type(node).__name__}')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Служебные имена запрещены.')
        if isinstance(node, ast.Attribute) and node.attr not in ('fromkeys', 'append', 'add'):
            raise ValueError(f'Недопустимый атрибут: {node.attr}')
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Attribute)
                or isinstance(node.func, ast.Name) and node.func.id in BUILTINS):
            raise ValueError('Недопустимый вызов функции.')
    return tree


def worker(source):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    tree = check_source(source)
    scope = {'__builtins__': BUILTINS}
    exec(compile(tree, '<dedupe>', 'exec'), scope)
    results = []
    for values, expected in CASES:
        # Ограничиваем число операций, в том числе для цикла, растящего свой список.
        budget = 10000
        def trace(frame, event, arg):
            nonlocal budget
            if frame.f_code.co_filename == '<dedupe>':
                frame.f_trace_opcodes = True
                if event == 'opcode':
                    budget -= 1
                    if budget <= 0:
                        raise RuntimeError('Превышен предел операций.')
            return trace
        sys.settrace(trace)
        try:
            actual = scope['dedupe'](values.copy())
        finally:
            sys.settrace(None)
        # В отчёт не сериализуем потенциально большой/циклический результат.
        valid_list = type(actual) is list and len(actual) <= 20 and all(type(v) is int for v in actual)
        results.append({'input': values, 'expected': expected,
                        'actual': actual if valid_list else 'не список целых длиной до 20',
                        'ok': valid_list and actual == expected})
    return {'ok': all(row['ok'] for row in results), 'cases': results}


def validate(source):
    try:
        run = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve())],
                             input=source, text=True, capture_output=True, timeout=3)
        if run.returncode:
            return {'ok': False, 'error': 'Процесс проверки завершился аварийно.'}
        return json.loads(run.stdout)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'Превышено время проверки.'}


if __name__ == '__main__':
    try:
        result = worker(sys.stdin.read())
    except (SyntaxError, ValueError, TypeError, NameError, AttributeError, RuntimeError) as exc:
        result = {'ok': False, 'error': str(exc)}
    print(json.dumps(result, ensure_ascii=False))
