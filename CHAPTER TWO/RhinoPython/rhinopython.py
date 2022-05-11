import rhinoscriptsyntax as rs
def sub(input, generation):
    if generation > 0:
        segs = rs.DivideCurve(input,3)
        cenVec = rs.VectorCreate(segs[2],segs[1])
        rotVec = rs.VectorRotate(cenVec,60,[0,0,-1])
        newPt = rs.VectorAdd(segs[1],rotVec)
        c1 = rs.AddLine(segs[0],segs[1])
        c2 = rs.AddLine(segs[1],newPt)
        c3 = rs.AddLine(newPt,segs[2])
        c4 = rs.AddLine(segs[2],segs[3])
        rs.DeleteObject(input)
        
        sub(c1,generation-1)
        sub(c2,generation-1)
        sub(c3,generation-1)
        sub(c4,generation-1)
crv = rs.GetObject()
sub(crv,4)