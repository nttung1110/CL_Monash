import numpy as np
import json

test_data = {'file_name': 'aa',
            'array': [[1,2,3,4], [5,6,7]]}

path_write = './test.json'
with open(path_write, 'w') as fp:
    json.dump(test_data, fp)