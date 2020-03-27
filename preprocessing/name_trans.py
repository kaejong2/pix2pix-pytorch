import numpy as np
import cv2
import os
import os.path



a = os.listdir('name/A')
b = os.listdir('name/B')
#
#
# for filename in os.listdir('name/A'):
#     print('name/A/'+filename, ' =>','name/A/'+filename[:22]+'.png')
#     os.rename('name/A/'+filename, 'name/A/'+filename[:22]+'.png')
#
num = 0
for i in range(len(a)):
    if a[i] == b[i]:
       num +=1

print(num)