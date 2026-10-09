"""教材共用小型數值檢查；純標準庫、合成資料、不控制設備。"""
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('向量維度不同')
    return sum(x*y for x,y in zip(a,b))


def transpose(a):
    if not a or not a[0] or any(len(row)!=len(a[0]) for row in a):
        raise ValueError('必須為非空矩形矩陣')
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    bt=transpose(b)
    transpose(a)
    if len(a[0])!=len(b):
        raise ValueError('矩陣維度不相容')
    return [[dot(row,col) for col in bt] for row in a]


def softmax(x):
    if not x or not all(math.isfinite(v) for v in x):
        raise ValueError('softmax需要有限非空輸入')
    ex=[math.exp(v-max(x)) for v in x]
    total=sum(ex)
    return [v/total for v in ex]


def solve(a, b):
    n=len(a)
    if n==0 or len(b)!=n or any(len(row)!=n for row in a):
        raise ValueError('只接受方陣及匹配向量')
    m=[list(map(float,row))+[float(value)] for row,value in zip(a,b)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda row:abs(m[row][col]))
        if abs(m[pivot][col])<1e-12:
            raise ValueError('奇異或近奇異矩陣；此教學實作不能替代成熟數值庫')
        m[col],m[pivot]=m[pivot],m[col]
        scale=m[col][col]
        m[col]=[v/scale for v in m[col]]
        for row in range(n):
            if row==col:continue
            factor=m[row][col]
            m[row]=[v-factor*w for v,w in zip(m[row],m[col])]
    return [row[-1] for row in m]


def projection(u, v):
    denom=dot(v,v)
    if denom==0:raise ValueError('不能投影到零向量方向')
    return [dot(u,v)/denom*x for x in v]


def attention(q,k,v):
    scores=matmul(q,transpose(k))
    scale=math.sqrt(len(q[0]))
    weights=[softmax([s/scale for s in row]) for row in scores]
    return weights,matmul(weights,v)


def self_test():
    assert matmul([[1,2],[3,4]],[[1],[2]])==[[5],[11]]
    x=solve([[2,1],[1,3]],[5,6])
    assert all(abs(a-b)<1e-10 for a,b in zip(x,[1.8,1.4]))
    assert projection([2,1],[1,0])==[2,0]
    sm=softmax([1000,1000]);assert sm==[.5,.5]
    weights,out=attention([[1,0],[0,1]],[[1,0],[0,1]],[[10],[20]])
    assert all(abs(sum(row)-1)<1e-12 for row in weights)
    assert 10<out[0][0]<15 and 15<out[1][0]<20
    # 線性層平方損失對w的梯度：2(wx-y)x；用中心有限差分核對。
    w,x,y=2.0,3.0,5.0
    loss=lambda value:(value*x-y)**2
    numerical=(loss(w+1e-6)-loss(w-1e-6))/(2e-6)
    assert abs(numerical-2*(w*x-y)*x)<1e-6
    try:solve([[1,2],[2,4]],[1,2])
    except ValueError:pass
    else:raise AssertionError('奇異矩陣應拒絕')
    # 檢索只產生文件排名，不把向量分數轉成現場控制指令。
    query=[1,0];documents=[[1,0],[0,1]]
    assert max(range(2),key=lambda i:dot(query,documents[i]))==0
    print('8項數值檢查通過；不代表生成章節的程式均已驗證。')


if __name__=='__main__':self_test()
