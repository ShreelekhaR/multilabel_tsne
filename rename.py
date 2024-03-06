import os
# rename each file in images folder to have a png extension
lines = open('img_paths.txt').read().splitlines()
new_file = open('img_paths1.txt', 'w')
for line in lines:
    nl =  line.replace('images', 'assets').replace('jpg', 'png')
    new_file.write(nl + '\n')

new_file.close()
