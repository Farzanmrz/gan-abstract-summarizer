# Imports
import numpy as np

file_path = "./cnn_stories_tokenized/0ab5929d6d3956f5d1a3645fc81adcbe301f1404.story"

# Open the file and read its contents
with open(file_path, 'r') as file:
    content = file.read()

# Print the content of the file
print(content)