def yolo2bb(folder,bbfile,image_shape):
    dh, dw = image_shape
    bbfile=bbfile.split(".")[0]+".txt"

    fl = open(folder+"/"+bbfile, 'r')
    data = fl.readlines()
    fl.close()

    t=len(data)
    L=[(0,0) for i in range(t)]
    R=[(0,0) for i in range(t)]
    
    i = 0
    for dt in data:

        # Split string to float
        _, x, y, w, h = map(float, dt.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        
        L[i]=(l, t)
        R[i]=(r,b)
        i = i + 1
    return L,R