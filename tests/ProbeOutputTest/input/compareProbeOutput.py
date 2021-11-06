#! /usr/bin/env python

import csv, sys

if len(sys.argv) < 3:
  print("compareProbeOutput.py requires two filename arguments", file=sys.stderr)
  sys.exit(1)
if len(sys.argv) > 3:
  print("compareProbeOutput.py requires two filename arguments, but received an extra argument.", \
        file=sys.stderr)

paths = list()
csvread = list()
for k in range(2):
  try:
    paths.append(open(sys.argv[k+1], 'r'))
  except FileNotFoundError:
    print("compareProbeOutput.py: \"{}\" file not found".format(paths[k]), file=sys.stderr)
    sys.exit(1)

  csvread.append(csv.reader(paths[k], delimiter=','))

numdiffs = 0
linenumber = 0
eofA = False
eofB = False

while not (eofA or eofB):
  linenumber += 1
  try:
    lineA = next(csvread[0]);
  except StopIteration:
    eofA = True
  try:
    lineB = next(csvread[1]);
  except StopIteration:
    eofB = True

  if (eofA and eofB):
    break

  if (eofA and not eofB):
    print("compareProbeOutput.py: file 1 ended before file 2, at line number {}".format(linenumber), \
          file=sys.stderr)
    numdiffs += 1
    break

  if (eofB and not eofA):
    print("compareProbeOutput.py: file 2 ended before file 1, at line number {}".format(linenumber), \
          file=sys.stderr)
    numdiffs += 1
    break

  if len(lineA) != len(lineB):
    print("compareProbeOutput.py: Line {} has different lengths: length {} in file 1; length {} in file 2".format( \
          linenumber, len(lineA), len(lineB)), file=sys.stderr)
    print("  File 1: {}, File 2: {}".format(lineA, lineB), file=sys.stderr)
    numdiffs += 1
    continue

  # skip blank lines; we require that blank lines be in the same places in each file.
  if len(lineA) == 0:
    assert(len(lineB) == 0)
    continue

  if lineA[0] != lineB[0]:
    print("Line {} has different time values: t={} in file 1, t={} in file 2".format( \
          linenumber, lineA[0], lineB[0]), file=sys.stderr)
    numdiffs += 1
    break

  valueA = float(lineA[3])
  valueB = float(lineB[3])
  discrep = abs(valueB - valueA)
  if discrep > 1e-6 * abs(valueA):
    print("Line {} has different probe values: {} in file 1, {} in file 2".format( \
          linenumber, valueA, valueB), file=sys.stderr)
    numdiffs += 1
    continue

if numdiffs > 0:
  print("Files \"{}\" and \"{}\" differ".format(paths[0].name, paths[1].name), file=sys.stderr)
  sys.exit(1)
else:
  sys.exit(0)
