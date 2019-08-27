import argparse
import json

def compare_dict(a, b):
    akey = list(a.keys())
    bkey = list(b.keys())
    if set(akey) != set(bkey):
        return False
    for k in akey:
        if isinstance(a[k], dict) ^ isinstance(b[k], dict):
            return False
        elif isinstance(a[k], dict):
            return compare_dict(a[k], b[k])
        else:
            return a[k] == b[k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json1", type=str)
    parser.add_argument("json2", type=str)
    args = parser.parse_args()

    with open(args.json1, 'r') as f:
        dict1 = json.load(f)
    with open(args.json2, 'r') as f:
        dict2 = json.load(f)

    print(compare_dict(dict1, dict2))
