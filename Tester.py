def extractNumberFrom(fileName):
    nr=0
    i=0
    
    while not fileName[i].isdigit() and i<len(fileName):
        i+=1
        
    while fileName[i].isdigit() and i<len(fileName):
        nr*=10
        nr+=int(fileName[i])
        i+=1
        
    return nr

print(extractNumberFrom('Panarama12.jpg'))