import csv
from collections import Counter
def mtype(i):
    if i==0: return 'pass'
    if i<14: return 'single'
    if i<26: return 'double'
    if i<37: return 'triple'
    if i<169: return 'full_house'
    if i<325: return 'bomb'
    if i<378: return 'straight'
    if i<429: return 'sister'
    if i<456: return 'triple_straight'
    if i<463: return 'triple_straight'
    return 'sister'
def rank_aux(i):
    t=mtype(i)
    if t=='single': return (i-1+3,0)
    if t=='double': return (i-14+3,0)
    if t=='triple': return (i-26+3,0)
    if t=='full_house':
        x=i-37; tr=x//11+3; rem=x%11; return (tr, rem+3 if rem<(tr-3) else rem+4)
    if t=='bomb':
        x=i-169; r=x//13+3; rem=x%13
        return (r, 0 if rem==0 else (rem+2 if rem+2<r else rem+3))
    return (None,None)
def cat(teach, az):
    tt,ta=mtype(teach),mtype(az)
    if tt=='pass' or ta=='pass':
        other= ta if tt=='pass' else tt
        if other=='bomb': return '2. bomb vs pass'
        return ('1. pass vs play (teacher passed, az played)' if tt=='pass'
                else '1. pass vs play (az passed, teacher played)')
    if tt=='bomb' or ta=='bomb':
        if tt=='bomb' and ta=='bomb':
            return '5. bomb: diff rank' if rank_aux(teach)[0]!=rank_aux(az)[0] else '5. bomb: diff aux/kicker'
        return '3. bomb vs other (non-pass) move'
    if tt==ta:
        if tt=='single': return '4. which single (diff rank)'
        if tt=='double': return '6. which double (diff rank)'
        if tt=='triple': return 'which triple (diff rank)'
        if tt=='full_house':
            return '5. full house: diff aux/pair' if rank_aux(teach)[0]==rank_aux(az)[0] else '5. full house: diff rank'
        return f'which {tt} (diff rank/len)'
    return f'diff combo type (teacher:{tt} -> az:{ta})'

rows=[(int(r['teacher']),int(r['az'])) for r in csv.DictReader(open('/tmp/agree_pairs.csv'))]
tot=len(rows); dis=[(t,a) for t,a in rows if t!=a]; nd=len(dis)
c=Counter(cat(t,a) for t,a in dis)
print(f"REAL decisions={tot}  disagreements={nd} ({nd/tot:.1%})\n")
print(f"{'category':46s} count  %of-dis  %of-all")
for k,n in c.most_common():
    print(f"{k:46s} {n:5d}  {n/nd:6.1%}  {n/tot:5.1%}")
# grouped summary
import re
g=Counter()
for k,n in c.items():
    if k.startswith('1.'): g['pass vs play']+=n
    elif k.startswith('2.'): g['bomb vs pass']+=n
    elif k.startswith('3.'): g['bomb vs other']+=n
    elif 'bomb:' in k: g['bomb vs bomb (rank/aux)']+=n
    elif 'full house' in k: g['full house (rank/aux)']+=n
    elif k.startswith('4.'): g['which single']+=n
    elif k.startswith('6.'): g['which double']+=n
    elif k.startswith('diff combo'): g['diff combo type']+=n
    else: g['which (other same-type)']+=n
print("\n--- grouped ---")
for k,n in g.most_common(): print(f"{k:30s} {n:5d}  {n/nd:5.1%} of disagreements")
