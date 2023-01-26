import sys
from pprint import pprint

with open(sys.argv[1]) as data:
  pprint(data.readlines())
