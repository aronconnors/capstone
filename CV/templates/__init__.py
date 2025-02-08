from pathlib import Path

# Define the path to the templates directory
TEMPLATES_DIR = Path(__file__).parent  # Gets the 'templates' directory path

# Get a list of all PNG files in the directory
TEMPLATES = {file.stem: file for file in TEMPLATES_DIR.glob("*.png")}

# Example: Prints {'template1': 'templates/template1.png', 'template2': 'templates/template2.png'}

