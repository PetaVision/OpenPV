# Written by Max Theiler, max.theiler@gmail.com
# 2/5/2016

# This script for parsing the output file timers.txt,
# and writing it as a csv file comprehensible to the
# pie chart tools created by Brian Broom-Peltz.

# It also includes in the comments a crude mini-tutorial
# on python's regex tools.


# "re" is the best Python library for handling regexes. Used here for the re.search function 
import re
import csv

file = open("timers.txt");
master_list = file.readlines();
 
# '\w' matches any alphabet character but not whitespaces
# '+' means "any number of times >= 1".
# so '\w+' will match whole words of any length.
# in the string "   foo_bar %&#&**(( bash", the matches would be "foo_bar" and "bash"
first_name_pattern = "\w+"


# (?=xxx) is a pattern that means "only match patterns preceded by the string 'xxx'"
# '\s' matches any whitespace character
# so '\s+\w+' means "match any amount of whitespace immediately followed by any amount of alphas"
# for the pattern \s+\w+\s+\w+, the following would be matches:
# "   hello   world    " --> "   hello   world"
# "cogito ergo sum     " --> " ergo sum"
second_name_raw = "(?<=total time in)\s+\w+\s+\w+"


# The $ character means the end of the string.
# '\w+$' means "match any amount of alphas immediately prior to the end of the string"
# " call me maybe" --> "maybe"
# "call me maybe " would not return any matches.
third_name_pattern = "\w+$"

# '\d' matches digits: 123456789
# The . is just a .
# "\d+.\d+" means "match any number of digits followed by a dot followed by any number of digits"
# The final $, as above, means matches can only occur at the very end of the string.
# "abc123.456" --> "123.456"
# "abcdefg123" would not return any matches
# "abcdef123." would not return any matches
time_pattern = "\d+.\d+$"


output_names = ['label']
output_values = ['count']

for line in master_list:
    # re.search takes a regex as the first arg, the string to search as the second.
    # It returns special regex objects, the .group() function extracts the first matching string
    # This line will return a string like "    words   words",
    # asuming it can find one after the substring "total time in"
    raw = re.search(second_name_raw,line).group()

    first_name = re.search(first_name_pattern,line).group()
    second_name = re.search(first_name_pattern,raw).group()
    third_name = re.search(third_name_pattern,raw).group()
    
    output_names.append(first_name + '_' + second_name + '_' + third_name)
    output_values.append(re.search(time_pattern,line).group())

outlist = zip(*[output_names,output_values])

with open('timers.csv', 'w') as outfile:
    a = csv.writer(outfile, delimiter=',')
    a.writerows(outlist)

        
    
