import os
a = 'output'
for folder in os.listdir(a):
    if(os.path.isdir(a+'\\'+folder)):
        for t in os.listdir(a+'\\'+folder):
            os.remove(a+'\\'+folder+'\\'+t)
    else:
        os.remove(a+'\\'+folder)