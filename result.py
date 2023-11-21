import re

file_path = '/Users/irenehsu/Documents/Data Mining/codalab_contest/test.txt'

content = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        content.append(line.strip())
       
data = []
for ele in range(len(content)):
    data.append(content[ele].split(':'))

file_path = '/Users/irenehsu/Documents/Data Mining/codalab_contest/10.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

with open('result.txt', 'w', encoding='utf-8') as file:
    for i in range(1, len(data)):

        match = re.search(re.escape(data[i][1]), text)

        if match:
            start_position = match.start()
            end_position = match.end()
        
        input = str(data[0][0]) + '\t' + str(data[i][0]) + '\t' + str(start_position) + '\t' + str(end_position) + '\t' + str(data[i][1] + '\n')
        file.write(input)