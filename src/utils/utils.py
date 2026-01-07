import importlib

def load_class(class_path, base_path="src.models"):
    """
    class_path: "my_model.MyModel"
    base_path: "src.models" 또는 "src.datasets"
    """
    module_path, class_name = class_path.rsplit(".", 1)
    full_module_path = f"{base_path}.{module_path}"
    
    module = importlib.import_module(full_module_path)
    return getattr(module, class_name)