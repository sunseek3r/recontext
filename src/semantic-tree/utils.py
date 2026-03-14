import re

def extract_imports_from_prefix(prefix_text):
    """
    Safely extracts imported modules and specific symbols from a FIM prefix string
    using regex, avoiding SyntaxErrors from incomplete code.
    """
    # Dictionaries to store our findings
    # standard_imports: tracks `import X` -> {"X": None} or {"X": "alias"}
    # from_imports: tracks `from X import Y` -> {"X": ["Y", "Z"]}
    extracted_data = {
        "standard_imports": {},
        "from_imports": {}
    }

    # Regex patterns for Python imports
    # Matches: import os, sys | import numpy as np
    regex_import = re.compile(r'^\s*import\s+(.+)', re.MULTILINE)
    
    # Matches: from typing import List, Dict | from .utils import auth
    regex_from = re.compile(r'^\s*from\s+([\w\.]+)\s+import\s+(.+)', re.MULTILINE)

    # 1. Process standard `import ...` statements
    for match in regex_import.finditer(prefix_text):
        modules_str = match.group(1)
        # Handle multiple modules separated by commas (e.g., import os, sys)
        for module_part in modules_str.split(','):
            module_part = module_part.strip()
            if ' as ' in module_part:
                # Handle aliases (import numpy as np)
                base_module, alias = module_part.split(' as ')
                extracted_data["standard_imports"][base_module.strip()] = alias.strip()
            else:
                extracted_data["standard_imports"][module_part] = None

    # 2. Process `from ... import ...` statements
    for match in regex_from.finditer(prefix_text):
        base_module = match.group(1).strip()
        imported_items_str = match.group(2)
        
        # Remove any inline comments that might be trailing
        imported_items_str = imported_items_str.split('#')[0]
        
        # Handle multiple imported items (from x import y, z)
        items = []
        for item in imported_items_str.split(','):
            item = item.strip()
            if item:
                # We can ignore 'as' aliases here for simplicity, 
                # or split on ' as ' if you want to track local aliases exactly.
                clean_item = item.split(' as ')[0].strip()
                items.append(clean_item)
                
        if base_module not in extracted_data["from_imports"]:
            extracted_data["from_imports"][base_module] = []
        extracted_data["from_imports"][base_module].extend(items)

    return extracted_data

# --- Example Usage ---
mock_prefix = """
import os, sys
import numpy as np
from core.database import DatabaseConnector, QueryBuilder
from .utils import format_string

def process_data(data):
    # Cursor is here...
"""

imports = extract_imports_from_prefix(mock_prefix)
print("Standard Imports:", imports["standard_imports"])
print("From Imports:", imports["from_imports"])