"""
Continual Learning Methods Registry
"""

class CLMethodRegistry:
    """Registry for continual learning methods"""
    _registry = {}
    
    @classmethod
    def register(cls, name):
        """
        Decorator to register a CL method
        
        Args:
            name: Name of the method
        
        Example:
            @cl_method_registry.register("spark_avn")
            class SparkAVN(BaseCLMethod):
                pass
        """
        def decorator(method_class):
            cls._registry[name.lower()] = method_class
            return method_class
        return decorator
    
    @classmethod
    def get(cls, name):
        """
        Get a registered CL method class
        
        Args:
            name: Name of the method
            
        Returns:
            The CL method class, or None if not found
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_methods(cls):
        """List all registered methods"""
        return list(cls._registry.keys())


# Global registry instance
cl_method_registry = CLMethodRegistry()

# Import all CL methods to trigger registration
from cavn.cl_method.base_cl_method import BaseCLMethod
from cavn.cl_method.finetune import Finetune
from cavn.cl_method.spark_avn import SparkAVN


__all__ = [
    'cl_method_registry',
    'BaseCLMethod',
    'Finetune',
    'SparkAVN',
]
