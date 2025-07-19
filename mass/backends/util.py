import inspect
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union, Optional, Dict, List
import ast
import operator
import math

def execute_function_calls(function_calls, tool_mapping):
        """Execute function calls and return formatted outputs for the conversation."""
        function_outputs = []
        for function_call in function_calls:
            try:
                # Get the function from tool mapping
                target_function = None
                function_name = function_call.get('name')
                
                # Look up function in tool_mapping
                if function_name in tool_mapping:
                    target_function = tool_mapping[function_name]
                else:
                    # Handle error case
                    error_output = {
                        "type": "function_call_output",
                        "call_id": function_call.get('call_id'),
                        "output": f"Error: Function '{function_name}' not found in tool mapping"
                    }
                    function_outputs.append(error_output)
                    continue
                
                # Parse arguments and execute function
                arguments = json.loads(function_call.get('arguments', '{}'))
                result = target_function(**arguments)
                
                # Format the output according to Responses API requirements
                function_output = {
                    "type": "function_call_output",
                    "call_id": function_call.get('call_id'),
                    "output": str(result)
                }
                function_outputs.append(function_output)
                
                print(f"Executed function: {function_name}({arguments}) -> {result}")
                
            except Exception as e:
                # Handle execution errors
                error_output = {
                    "type": "function_call_output", 
                    "call_id": function_call.get('call_id'),
                    "output": f"Error executing function: {str(e)}"
                }
                function_outputs.append(error_output)
                print(f"Error executing function {function_name}: {e}")
                
        return function_outputs


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}")

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}")
        parameters[param.name] = {"type": param_type}

    required = [param.name for param in signature.parameters.values() if param.default == inspect._empty]

    return {
        "type": "function",
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
        },
    }