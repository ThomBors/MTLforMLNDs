#!/usr/bin/env python

import importlib
import sys
import os

def dynamic_import_and_run(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        function()
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
    except AttributeError as e:
        print(f"Error accessing {function_name} in {module_name}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Add the current directory to sys.path to make sure submodules can be found
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    #dynamic_import_and_run('STL.run_experiments', 'mainSTL')
    # dynamic_import_and_run('MTL.run_experimentsCobMTL', 'mainMTL')
    dynamic_import_and_run('MTL.run_resultsMTL','mainMTL')
    

if __name__ == "__main__":
    main()