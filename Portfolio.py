import numpy
import sys
import math
import pickle
import os
import urllib
import datetime
from scipy.stats import norm
from cvxopt import matrix
from cvxopt import solvers
import shutil

base = "BA"

PC = {}
globals()["riskmodel"] = "s" #d = downside s = sharpes m=compared to market(QQQ)
class Portfolio:
    def __init__(self):
        self.lib = {}
    def add(self,sym,shares,purchase):
        self.lib[sym]=[shares,purchase]
    def allocation(self):
        total = self.eval()
        retr = datetime.datetime.now() + datetime.timedelta(-1)
        
        alloc = []
        for i in self.lib:
           p = GetPrice(i,str(retr.month),str(retr.day),str(retr.year))
           a = 100*self.lib[i][0]*p/total
           alloc.append(a)
        return alloc
    def realloc(self,dist):
        retr = datetime.datetime.now() + datetime.timedelta(-1)
        orders = []
        total = self.eval()
        alloc = self.allocation()
        
        i = 0
        while i<len(self.lib):
           p = GetPrice(self.lib.keys()[i],str(retr.month),str(retr.day),str(retr.year))
           bs = "S"
           if float(dist[i])-alloc[i] > 0:
              bs = "B"
           orders.append([self.lib.keys()[i],bs,abs((float(dist[i])-alloc[i])*total/100/p)])
           i+=1
        
        return orders
    def list(self):
        #print self.lib.keys()
        self.sigma = AllCov(self.lib.keys())
        #print "sigma:",self.sigma
        self.cv = 2*self.sigma
        return self.lib.keys()
    def load(self,name):
        pass
    def loadlist(self):
        self.lib = {}
        port = open("./Portfolios/test.txt","r")
        for i in port:
            i = i.replace("\n","")
            i = i.replace("\r","")
            if i.find(",")>0:
                i = i[:i.find(",")]
            self.add(i,0,0)
        return self.lib.keys()
    def save(self,name):
        pass
    def eval(self):
        retr = datetime.datetime.now() + datetime.timedelta(-1)
        total = 0
        for i in self.lib:
            p = GetPrice(i,str(retr.month),str(retr.day),str(retr.year))
            #print i,self.lib[i],p,self.lib[i][0]*p
            total += self.lib[i][0]*p
        return total
    def evalasof(self,y,m,d):
        total = 0
        for i in self.lib:
            p = GetPrice(i,str(m),str(d),str(y))
            #print i,self.lib[i],p,self.lib[i][0]*p
            total += self.lib[i][0]*p
        return total
    def efficient(self):
        pass
    def tangency(self):
        pass

    def targetreturn(self,p,r,strat="N"):
        
        r = float(r)/12.0
        #print "S1:",self.sigma
        a,b,d,e = "","","",""
        if strat == "N":
           a,b,d,e = self.test(p)
        else:
           #CURRENTLY FAILS TO GIVE VALID OUTPUT
           a,b,d,e = self.noshort(p)
        #print "S2:",d

        i = 0
        while i < len(e):
            e[i]/=1200
            i+=1

        g = numpy.hstack((d,numpy.transpose(numpy.matrix(e))))
        g = numpy.hstack((g,numpy.transpose(numpy.matrix([1 for i in range(len(d))]))))
        g = numpy.vstack((g,numpy.matrix(e+[0,0])))
        g = numpy.vstack((g,numpy.matrix([1 for i in range(len(d))]+[0,0])))

        bvec = numpy.transpose(numpy.matrix(([0 for i in range(len(d))]+[r,1])))
        solu=numpy.linalg.solve(g,bvec)

        pct = "["
    
        ct = 1
        totalpospct,totalnegpct=0,0
        for i in solu:
	    pct += "[" + str(float(i)) + "],"
            if float(i)<0:
                totalnegpct+=float(i)
            else:
                totalpospct+=float(i)
            if ct == len(solu)-2:
                break
            ct +=1
        pct = pct[:-1]+"]"

        yvec=numpy.matrix(pct)
        print "yvec:",yvec

        self.sd = yvec*d*numpy.transpose(yvec)
        self.ret = r*12

        print totalpospct,totalnegpct
                   
        return solu,g,bvec,yvec
    def noshort(self,t):
        r=[]
        for i in t:
            r.append(LookupReturn(i))
        Q = matrix(self.cv)
        p = matrix([0.0]*len(t))
        A = matrix([1.0]*len(t),(1,len(t)))
        b = matrix(1.0)
        G = matrix(numpy.identity(len(t)))
        h = matrix([0.0]*len(t))

        sol = solvers.qp(Q,p,-G,-h,A,b)
        
        ret,i = 0,0
        pcts = "["
        while i < len(sol['x']):
           pcov = PairCov(t[i],t[i],PC)[1]
           rt = round(r[i],2)
           print str(round(float(sol['x'][i]),4)).rjust(5),t[i].rjust(4) + " Ret",rt, "SD",pcov,"5%,1% VaR on $100K:",round(100000*norm.ppf(0.05,rt/1200.0,pcov),2),round(norm.ppf(0.01,rt/1200.0,pcov)*100000,2)
           pcts+="["+str(round(float(sol['x'][i]),4))+"],"
           ret += float(r[i])/100.0*round(float(sol['x'][i]),4)
           i += 1
        pcts = pcts[:-1] + "]"

        yvec = numpy.transpose(numpy.matrix(pcts))
        sd = numpy.transpose(yvec)*self.sigma*yvec
        self.ret=ret
        self.sd=sd
        return ret,yvec,self.sigma,r

    def test(self,t):
        r=[]
        for i in t:
            r.append(LookupReturn(i))
        tpospct, tnegpct = 0.0,0.0
        #CALC MIN VARIANCE PORTFOLIO
        if len(self.cv)>0 and str(self.cv[0][0])!="nan":
        
            newcol = ",[1]" * (len(t)-1)
            newcol = "[[1]"+str(newcol)+"]"
            newcol = numpy.transpose(numpy.matrix(newcol))
            self.cv = numpy.hstack((self.cv,newcol))
        
            newrow = numpy.matrix("["+"1,"*len(t)+ "0]")
        
            self.cv = numpy.vstack((self.cv,newrow))
            bcol="[0],"*(len(t))+"[1]"
            b=numpy.transpose(numpy.matrix(bcol))

            solu=numpy.linalg.solve(self.cv,b)

    	    #CALC PORTFOLIO RETURN
            i,ret = 0,0
            pcts = "["
            pctss = []
            while i < len(t):
                pcov = PairCov(t[i],t[i],PC)[1]
                rt = round(r[i],2)
                print str(round(100*float(solu[i][0]),2)).rjust(5),t[i].rjust(4) + " Ret",rt, "SD",pcov,"5%,1% VaR on $100K:",round(100000*norm.ppf(0.05,rt/1200.0,pcov),2),round(norm.ppf(0.01,rt/1200.0,pcov)*100000,2)
            
                ret += float(r[i])/100.0*float(solu[i][0])
                pcts+="["+str(float(solu[i][0]))+"],"
                pctss.append(float(solu[i][0]))
                i += 1

                if round(float(solu[i][0]),3) > 0:
                    tpospct += round(float(solu[i][0]),3)
                else:
                    tnegpct += round(float(solu[i][0]),3)
     
            pcts = pcts[:-1] + "]"

            yvec = numpy.transpose(numpy.matrix(pcts))
            sd = numpy.transpose(yvec)*self.sigma*yvec
        
            print tpospct,tnegpct
        #CALC PORTFOLIO SHARPE RATIO
    
        #APPEND TO LIST OF SHARPE RATIOS
            self.ret=ret
            self.sd=sd
        return ret,yvec,self.sigma,r

    def plot(self):
        pass

    def summarize(self):

          print "Ret:",self.ret*100,"SD:",math.sqrt(abs(self.sd)),"Sharpe:",float(self.ret/12)/math.sqrt(abs(self.sd))
          print "5%,1% VaR on $100000:",round(100000*norm.ppf(0.05,self.ret/12.0,math.sqrt(abs(self.sd))),2),round(100000*norm.ppf(0.01,self.ret/12.0,math.sqrt(abs(self.sd))),2)
        

        
def IsFloat(f):
    try:
       float(f)
    except:
       return False
    return True

def LookupReturn(l):
    bret = False
    g = open("nasd_output_monthly.txt","r")

    for ln in g:
        if ln[:ln.find(",")]==l:      
            bret = ln[ln.rfind(",")+1:]   
            break
    g.close()
    if bret == False:
        g = open("nyse_output_monthly.txt","r")
        for ln in g:
           if ln[:ln.find(",")]==l:      
              bret = ln[ln.rfind(",")+1:]   
              break
        g.close()

    return float(bret)


def ReturnPoint(p):
    r = LookupReturn(p)
    x,y,z = PairCov(p,p,PC)
    return r,y

def PairCov(t1,t2,PC,riskmodel="s"):
    #print "RISK:",riskmodel
    if t1>t2:
       t3=t1
       t1=t2
       t2=t3
    
    if (t1,t2) in PC:
        return PC[t1,t2]

    
    try:
       f = open("./monthly/"+t1+".txt","r")
    except:
       return "",0,0
    st1,st2,stq=[],[],[]
    downside1,downside2,downsideq=[],[],[]

    q = open("./monthly/QQQ.txt","r")
    chgq=0
    oldstq=""
    ct = 0
    for ln in f:
        ct +=1
        #print "T2:",t2
        try:
           g = open("./monthly/"+t2+".txt","r")
        except:
           return "",0,0

        try:
           q = open("./monthly/QQQ.txt","r")
        except:
           return "",0,0

        for ln3 in q:
            #print ln3[:ln3.find(",")],ln[:ln.find(",")]
            if ln3[:ln3.find(",")] == ln[:ln.find(",")] and ct != 1:
                #print ln3[:ln3.find(",")],ln[:ln.find(",")],ct
                if ct!=2:
                    chgq = ((oldstq / float(ln3[ln3.rfind(",")+1:]))-1)
                    #print "chq:",chq
                    stq.append(chgq)
                    if chgq<0:
                        downsideq.append(chgq)
                    else:
                        downsideq.append(0)
                    if chgq<0:
                        downsideq.append(chgq)
                    else:
                        downsideq.append(0)
                oldstq = float(ln3[ln3.rfind(",")+1:])
                break
        
        for ln2 in g:
            if ln2[:ln2.find(",")] == ln[:ln.find(",")] and ct != 1:
                if ct!=2:
                    chg1 = ((oldst1 / float(ln[ln.rfind(",")+1:]))-1)
                    chg2 = ((oldst2 / float(ln2[ln2.rfind(",")+1:]))-1)
                    
                    if riskmodel != "m":
                        st1.append(chg1)
                        st2.append(chg2)
                        #print "a",riskmodel,st1[0]
                    else:
                        st1.append(chg1-chgq)
                        st2.append(chg2-chgq)
                        chg1 = chg1-chgq
                        chg2 = chg2-chgq 
                        #print "b",riskmodel,st1[0],chgq
                    if chg1<0:
                        downside1.append(chg1)
                    else:
                        downside1.append(0)
                    if chg2<0:
                        downside2.append(chg2)
                    else:
                        downside2.append(0)
                try:
                   oldst1 = float(ln[ln.rfind(",")+1:])
                   oldst2 = float(ln2[ln2.rfind(",")+1:])
                except:
                   return "",0,0

                break
        g.close()

    if riskmodel=="s":
        PC[t1,t2]=numpy.cov(st1,st2),numpy.std(st1),numpy.std(st2)
        return numpy.cov(st1,st2),numpy.std(st1),numpy.std(st2)
    elif riskmodel =="d":
        PC[t1,t2]=numpy.cov(downside1,downside2),numpy.std(downside1),numpy.std(downside2)     
        return numpy.cov(downside1,downside2),numpy.std(downside1),numpy.std(downside2)
    elif riskmodel == "m":
        PC[t1,t2]=numpy.cov(st1,st2),numpy.std(st1),numpy.std(st2)    
        return numpy.cov(st1,st2),numpy.std(st1),numpy.std(st2)


def AllCov(n):
    PC={}
    #print "RISK1:",riskmodel
    if os.path.exists("pickle_"+riskmodel+".dmp") and len(n)>25:
        dstr = open("pickle_"+riskmodel+".dmp","r")
        PC = pickle.load(dstr)
        dstr.close()
    row = []
    totalpairs = 0

    i=1
    while i<=len(n):
        row.append(0.0)
        totalpairs += i
        i +=1
        
    i=0
    col = []
    while i<len(n):
       col.append(row)
       i+=1
    
    sigma = numpy.matrix(col)
    orig_n=len(n)

    ct = 0
    for x in range(len(n)):
       
       #print x
       for y in range(len(n)):
          ct += 1
          #print x,n[x],y,n[y],ct
          
          o,p,q = PairCov(n[x],n[y],PC,riskmodel)

          if x==y:
             sigma[[x],[x]]=float(p)*float(p)
          else:
             sigma[[x],[y]]=float(o[0][1])

          if len(n)>25 and ct % 72500 == 0 and ct > 797000:
             try:
                shutil.copy2("pickle_"+riskmodel+".dmp", "pickle_"+riskmodel+".bak")
             except:
                pass
             dser = open("pickle_"+riskmodel+".dmp","w")
             pickle.dump(PC,dser)
             dser.close()
 
    if len(n)>25:
       try:
          shutil.copy2("pickle_"+riskmodel+".dmp", "pickle_"+riskmodel+".bak")
       except:
          pass
       dser = open("pickle_"+riskmodel+".dmp","w")
       pickle.dump(PC,dser)
       dser.close()
 
    return sigma



def EvalPortfolio(t):
    rets,eqs = [],{}

    ret,pct,sigma,r=TestPortfolio(t)

    if ret:
       rets.append(ret)
       eqs[ret]=[t[0],pct]
    
    return sigma

def Calculate(sigma,vec,t):
    pts=[]
    ret,ct = 0,0

    for i in vec:
        ret+=float(LookupReturn(t[ct]))/100.0*float(vec[ct])
        ct+=1
   
    xvec = numpy.matrix(vec)
    sd = numpy.transpose(xvec)*sigma*xvec

    if sd < 0:
        sd = -math.sqrt(abs(sd))
    else:
        sd = math.sqrt(sd)
    
    pts.append((ret,float(sd)))

    return pts

def EnumeratePortfolio(t):
     pts=[]
     sigma = EvalPortfolio(t)
     b,s = EfficientPortfolio(t)

     for i in range(200):
         vec = b + (i * s)
         pts.append(Calculate(sigma,vec,t))
         if i != 0:
             vec = b + (-i * s)
             pts.append(Calculate(sigma,vec,t))
         
     return pts


def LTestPortfolio(riskmodel="s"):
    #FULLY DEPRICATED
    p = []  
    port = open("./Portfolios/test.txt","r")
    for i in port:
        i = i.replace("\n","")
        i = i.replace("\r","")
        if i.find(",")>0:
            i = i[:i.find(",")]
        p.append(i)

    return TestPortfolio(p,riskmodel)


def EfficientPortfolio(t):
    a,b,d,e = TestPortfolio(t)
    
    g,h,i = TargetReturn(t,a+0.01)
    scalar = g[0:len(t)]-b
    
    return b,scalar

def TestPortfolio(t,riskmodel="s"):
    globals()["riskmodel"] = riskmodel
    #FULLY DEPRICATED
    r=[]
    for i in t:
        r.append(LookupReturn(i))
    
    mr = r[0]       
    ret,pct="",""
    
    #CALC COVARIANCE
    sigma = AllCov(t)

    cv = 2*sigma

    tpospct, tnegpct = 0.0,0.0
    #CALC MIN VARIANCE PORTFOLIO
    if len(cv)>0 and str(cv[0][0])!="nan":
        
        newcol = ",[1]" * (len(t)-1)
        newcol = "[[1]"+str(newcol)+"]"
        newcol = numpy.transpose(numpy.matrix(newcol))
        cv = numpy.hstack((cv,newcol))
        
        newrow = numpy.matrix("["+"1,"*len(t)+ "0]")
        
        cv = numpy.vstack((cv,newrow))
        bcol="[0],"*(len(t))+"[1]"
        b=numpy.transpose(numpy.matrix(bcol))

        solu=numpy.linalg.solve(cv,b)

    	#CALC PORTFOLIO RETURN
        i = 0
        ret = 0
        pcts = "["
        pctss = []
        while i < len(t):
            pcov = PairCov(t[i],t[i],PC,riskmodel)[1]
            rt = round(r[i],2)
            print str(round(100*float(solu[i][0]),2)).rjust(5),t[i].rjust(4) + " Ret",rt, "SD",pcov,"5%,1% VaR on $100K:",round(100000*norm.ppf(0.05,rt/1200.0,pcov),2),round(norm.ppf(0.01,rt/1200.0,pcov)*100000,2)
            
            ret += float(r[i])/100.0*float(solu[i][0])
            pcts+="["+str(float(solu[i][0]))+"],"
            pctss.append(float(solu[i][0]))
            i += 1

            if round(float(solu[i][0]),3) > 0:
                tpospct += round(float(solu[i][0]),3)
            else:
                tnegpct += round(float(solu[i][0]),3)
     
        pcts = pcts[:-1] + "]"
        yvec = numpy.transpose(numpy.matrix(pcts))
        sd = numpy.transpose(yvec)*sigma*yvec
        if sd == 0:
           sd = 0.000001
        print "Ret:",ret*100,"SD:",math.sqrt(abs(sd)),"Sharpe:",float(ret/12)/math.sqrt(abs(sd))
        print "5%,1% VaR on $100000:",round(100000*norm.ppf(0.05,ret/12.0,math.sqrt(abs(sd))),2),round(100000*norm.ppf(0.01,ret/12.0,math.sqrt(abs(sd))),2)
        print tpospct,tnegpct
    #CALC PORTFOLIO SHARPE RATIO
    
    #APPEND TO LIST OF SHARPE RATIOS
    #print "RISK:",riskmodel
    return ret,yvec,sigma,r


def TargetReturn(p,r,riskmodel="s"):
    globals()["riskmodel"] = riskmodel
    r = float(r)/12.0

    a,b,d,e = TestPortfolio(p,riskmodel)

    i = 0
    while i < len(e):
        e[i]/=1200
        i+=1

    g = numpy.hstack((d,numpy.transpose(numpy.matrix(e))))
    g = numpy.hstack((g,numpy.transpose(numpy.matrix([1] *len(d)))))
    g = numpy.vstack((g,numpy.matrix(e+[0,0])))
    g = numpy.vstack((g,numpy.matrix([1] * len(d)+[0,0])))

    #CHANGED r,1 to r,0 to attempt 0 bias portfolio
    bvec = numpy.transpose(numpy.matrix(([0]*len(d)+[r,1])))
    solu=numpy.linalg.solve(g,bvec)

    pct = "["
    
    ct = 1
    totalpospct,totalnegpct=0,0
    for i in solu:
        print round(float(i)*100.0,2)
	pct += "[" + str(float(i)) + "],"
        if float(i)<0:
            totalnegpct+=float(i)
        else:
            totalpospct+=float(i)
        if ct == len(solu)-2:
           break
        ct +=1
    pct = pct[:-1]+"]"

    yvec=numpy.matrix(pct)

    sd = math.sqrt(abs(yvec*d*numpy.transpose(yvec)))
    print "RET:",r*1200,"s.d",sd,"Sharpe:",float(r)/sd
    print "5%,1% VaR on $100000:",round(100000*norm.ppf(0.05,r,sd),2),round(100000*norm.ppf(0.01,r,sd),2)
    print totalpospct,totalnegpct
                   
    return solu,g,bvec

def LoadTargetReturn(r,riskmodel="s"):
    globals()["riskmodel"] = riskmodel
    p=[]
    z = open(".\\portfolios\\test.txt","r")
    for ln in z:
        if ln[-1]=="\r\n" or ln[-1]=="\n":
           ln = ln[:-1]
        if ln.find(",")>0:
           ln = ln[:ln.find(",")]
        p.append(ln)
        
    return TargetReturn(p,r,riskmodel)

def LTargetReturn(l,r,riskmodel="s"):
    globals()["riskmodel"] = riskmodel
    p=[]
    z = open(".\\portfolios\\"+l+".txt","r")
    for ln in z:
        if ln[-1]=="\r\n" or ln[-1]=="\n":
           ln = ln[:-1]
        if ln.find(",")>0:
           ln = ln[:ln.find(",")]
        p.append(ln)
        
    return TargetReturn(p,r,riskmodel)

def EvaluatePortfolio(p,r):
    startdir = "./daily/2012-10-31/"
    stopdir = "./daily/2012-11-01/"
    
    Investment = 100000
    solu,g,bvec=TargetReturn(p,r)

    i,total = 0,0
    while i < len(p):
        start = open(startdir+p[i]+".txt","r")
        stop = open(stopdir+p[i]+".txt","r")

        #BEGIN DATE
        st,stp = "",""
        ct = 0
        for ln in start:
            if ct==1:
                print "start:",ln[ln.rfind(",")+1:]
                st = float(ln[ln.rfind(",")+1:])
            ct+=1
        ct = 0
        for ln in stop:
            if ct==1:
                print "stop:",ln[ln.rfind(",")+1:]
                stp = float(ln[ln.rfind(",")+1:])
            ct+=1

        chg = stp-st
        shares = Investment*float(solu[i])/st
        print str(round(float(solu[i]),4))+"%",p[i],chg,shares
        #ALLOCATE SHARES
        #PRINT TOTAL

        #END DATE
        #CALC DIFFERENCES
        #PRINT TOTAL, ANNUALIZED PROFIT/LOSS    
        total+=chg*shares
        i += 1

    print "Total:",total

def GetPrice(symbol,m,d,y):
    #IF PRICE NOT AVAILABLE IN CACHE - GRAB FROM YAHOO
    url = 'http://ichart.finance.yahoo.com/table.csv?s='+symbol+'&a='+str(int(m)-1)+'&b='+d+'&c='+y+'&d='+m+'&e='+d+'&f='+y+'&g=d&ignore=.csv'

    f = urllib.urlopen(url)
    s = f.read()
    f.close

    ln = ""
    lines = []
    for ch in s:
        if ch != '\n':
            ln += ch
        else:
            lines.append(ln)
            ln = ""

    return float(lines[1][lines[1].rfind(",")+1:])
    

def LAllocatePortfolio(r):
    p = []
    port = open("./Portfolios/test.txt","r")
    for i in port:
        i = i.replace("\n","")
        i = i.replace("\r","")
        p.append(i)

    h = open("holding.txt","w")
    Investment = 100000
    solu,g,bvec=TargetReturn(p,r)

    
    i,total,totalpct,totalpospct,totalnegpct = 0,0,0,0,0
    while i < len(p):
        #BEGIN DATE
        st= GetPrice(p[i],"10","31","2012")
        ret=float(solu[i])
        shares = round(Investment*ret/st,2)
        
        print str(round(ret*100,1))+"%",p[i],shares
        bs = "BUY"
        if shares<0:
           bs="SELL"
        csvstr = ",".join([p[i],bs,str(abs(shares)),"11-02-2012",str(st)])
        h.write(csvstr+"\r\n")
        #ALLOCATE SHARES
        #PRINT TOTAL
        #END DATE
        #CALC DIFFERENCES
        #PRINT TOTAL, ANNUALIZED PROFIT/LOSS
        totalpct += ret
        if ret > 0:
           totalpospct += ret 
        else:
           totalnegpct += ret

        i += 1

    h.close()

    print "Total:",total
    print "TotalPct:",totalpct,totalpospct,totalnegpct


    
        

    