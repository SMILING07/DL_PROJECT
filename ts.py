digits = "23"
hm  = {
    '2':'abc',
    '3':'def',
    '4':'ghi',
    '5':'jkl',
    '6':'mno',
    '7':'pqrs',
    '8':'tuv',
    '9':'wxyz'
}
n = len(digits)
prdt = 1
for x in range(len(digits)):
    if digits[x]=='7' or digits[x]=='9':
        prdt *=4
    else:
        prdt *=3

l=['']*prdt
def bt(digits,rp,k,start):
    if k>=n :
        return ''
    t = 4 if digits[k]=='7' or digits[k]=='9' else 3
    for i in range(t):
        p = hm[digits[k]][i]
        for j in range(start,start+t):
            l[j] += p
        print(l)
        l[start+i]+=rp+bt(digits,p,k+1,start+(t)*i)

    return ''
bt(digits,'',0,0)
print(l)
