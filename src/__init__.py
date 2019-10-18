import os, sys

#'/home/user/example/parent/child'
current_path = os.path.abspath('.')

#'/home/user/example/parent'
parent_path = os.path.dirname(current_path)

sys.path.append(parent_path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
