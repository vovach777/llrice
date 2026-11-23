import glob,os,sys
D=sys.argv[1]; G={}
for f in glob.glob(D+'/*.txt'):
    s=os.path.basename(f)[:-4]
    for l in open(f):
        t=l.split(); PS,z,zy,zs,y0,s0,y1,s1=map(int,t[1:9]); G.setdefault(s,[]).append((z/(3*PS),zy/PS,zs/PS,(y0,y1),(s0,s1)))
def tot(R,fy,fs): return sum(r[3][fy(r)]+r[4][fs(r)] for r in R)
sets=sorted(G)
ora=lambda R: tot(R,lambda r:int(r[3][1]<r[3][0]),lambda r:int(r[4][1]<r[4][0]))
def vals(fy,fs): return [100*(tot(G[s],fy,fs)/ora(G[s])-1) for s in sets]
print('%-26s'%'loss vs per-lane oracle %'+''.join('%9s'%s for s in sets))
def row(name,fy,fs):
    v=vals(fy,fs); print('%-26s'%name+''.join('%9.2f'%x for x in v)+'  mean %.3f'%(sum(v)/len(v)))
cur=lambda r:int(r[0]>=.65); bo=lambda r:int(r[3][1]+r[4][1]<r[3][0]+r[4][0])
row('R8F5 now (band 65%)',cur,cur); row('band oracle (1 flag)',bo,bo)
row('per-lane 65/65',lambda r:int(r[1]>=.65),lambda r:int(r[2]>=.65))
row('per-lane 50/50',lambda r:int(r[1]>=.5),lambda r:int(r[2]>=.5))
res=sorted((sum(vals(lambda r,a=ty:int(r[1]>=a/100),lambda r,b=ts:int(r[2]>=b/100)))/len(sets),ty,ts) for ty in range(30,100,5) for ts in range(30,100,5))
ty,ts=res[0][1],res[0][2]; row('per-lane best Y>=%d S>=%d'%(ty,ts),lambda r:int(r[1]>=ty/100),lambda r:int(r[2]>=ts/100))
cnt={}
for s in ['scans','full12','valid41']:
    c=[0]*4
    for r in G[s]: c[int(r[3][1]<r[3][0])|2*int(r[4][1]<r[4][0])]+=1
    print(s,'oracle Y/S: photo/photo %d screen/photo %d photo/screen %d screen/screen %d'%tuple(c))
