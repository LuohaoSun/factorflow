import inspect
import pkgutil
import importlib
import os
import factorflow
from factorflow.base import Selector

def test_selector_test_coverage_integrity():
    """检查是否所有非抽象的 Selector 子类都在测试代码中被引用."""
    
    # 1. 动态发现 factorflow 包中所有的 Selector 子类
    all_selectors = set()
    
    # 遍历 factorflow 下的所有模块
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        factorflow.__path__, factorflow.__name__ + "."
    ):
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 条件：是 Selector 的子类，不是 Selector 本身，且定义在 factorflow 包内
                if (
                    issubclass(obj, Selector) and 
                    obj is not Selector and 
                    obj.__module__.startswith("factorflow")
                ):
                    # 排除抽象基类 (含有 abstractmethod)
                    if not inspect.isabstract(obj):
                        all_selectors.add(obj.__name__)
        except ImportError:
            continue

    # 2. 读取 tests/ 目录下所有的测试文件内容
    test_dir = os.path.dirname(os.path.dirname(__file__)) # 指向 tests/
    test_codes = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # 排除本测试文件自身，避免自我循环引用
                if file == "test_selector_integrity.py":
                    continue
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    test_codes.append(f.read())
    
    combined_test_code = "\n".join(test_codes)

    # 3. 校验每个 Selector 是否在测试代码中出现过 (至少被 import 或提到)
    missing_selectors = []
    for selector_name in sorted(list(all_selectors)):
        if selector_name not in combined_test_code:
            missing_selectors.append(selector_name)

    assert not missing_selectors, (
        f"发现以下 Selector 类没有对应的测试用例（未在 tests/ 目录下的测试文件中被引用）: "
        f"{missing_selectors}. \n请为新开发的 Selector 编写单元测试！"
    )
