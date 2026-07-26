import re

with open("benchmark_script.py", "r") as f:
    content = f.read()

# Fix axes array indexing for Seaborn
content = content.replace("ax=axes)", "ax=axes[1])")
content = content.replace("ax=axes.", "ax=axes[1].")

with open("benchmark_script.py", "w") as f:
    f.write(content)
