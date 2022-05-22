from scipy.stats import norm
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import chi2_contingency

#Problem Statement 1
print("#Problem Statement 1")

print(" 1. Yes, as null hypthesis and alternate hypothesis are exactly opposite of each other")
print("2. No , as alternate hypothesis does not cover all the possible senarios in case null hypothesis is to be rejected")
print("3 No, because hypothesis is stated in terms of statistics and not sample data")
print("4 No, because values in both hypothesis is different and has equal sign")
print("5 No, because hypothesis are always statements about population or distribution and not about sample")

#Problem Statement 2
print("#Problem Statement 2")
p_mean = 52
p_std = 4.50
n = 100
sample_mean = 52.80
SE = p_std/n**0.5
Z = (sample_mean-p_mean)/SE
print(f"Z score is:{Z}")
alpha=0.05  #test_significance
print(f"Critical region is {norm.ppf(alpha/2)}, {-norm.ppf(alpha/2)}")
print("We can accept null hypothesis as z_score<critical region")

#Problem Statement 3
print("#Problem Statement 3")
p_mean = 34
p_std = 8
n = 50
sample_mean = 32.5
SE = p_std/n**0.5   #standard Error
Z = (sample_mean-p_mean)/SE
print(f"Z score is:{Z}")
alpha=0.01
print(f"Critical region is {norm.ppf(alpha/2)}, {-norm.ppf(alpha/2)}")
print("We can accept null hypothesis as z_score>critical region")

#Problem Statement 4
print("#Problem Statement 4")
given_data=[1008, 812, 1117, 1323, 1308, 1415, 831, 1021, 1287, 851, 930, 730, 699, 872, 913, 944, 954, 987, 1695, 995, 1003, 994]
p_mean =1135
sample_std = np.std(given_data)
n=22
sample_mean = np.sum(given_data,axis=0)/len(given_data)
SE = sample_std/n**0.5
alpha = 0.5
test_1 = (sample_mean-p_mean)/SE
print(f"t_Score is{test_1}")
print(f"Critical Region is {stats.t.ppf((alpha/2),df=21)} {stats.t.ppf(1-(alpha/2),df=21)}")
print("We can reject null hypothesis at alpha = 0.5  as it lies within critical region")

#Problem Statement 5
print("#Problem Statement 5")
p_mean = 48432
p_std = 2000
n =400
sample_mean =48574
SE = p_std/n**0.5
Z = (sample_mean-p_mean)/SE
print(f"Z score is:{Z}")
alpha=0.05
print(f"Critical region is {norm.ppf(alpha/2)} {-norm.ppf(alpha/2)}")
print("We can accept null hypothesis at alpha =0.05  as z_score is not in critical region")

#Problem Statement 6
print("#Problem Statement 6")
p_mean =32.28
n=19
sample_mean =31.67
sample_std =1.29
alpha =0.05
SE=sample_std/(n**0.5)
t=(sample_mean-p_mean)/SE
print(f"t_score is {round((t),1)}")
print(f"Critical region is {round(stats.t.ppf((alpha/2),df=18),1)} {-round(stats.t.ppf((alpha/2),df=18),1)}")
print("We can reject null hypothesis at alpha =0.05  asit lies in critical region")

#Problem Statement 7
print("#Problem Statement 7")
#Acceptance region 48.5 < x < 51.5
print("#Acceptance region 48.5 < x < 51.5")
# Calculate Beta at Mu1 = 52
n1 = 10
Sig =2.5
Mu1 =52
p11 = (48.5 - Mu1)/(Sig/math.sqrt(n1))
print(p11)
p12 = (51.5 - Mu1)/(Sig/math.sqrt(n1))
print(p12)
# As p12 > p11, so our z score lies in between these two p12 < z < p11.
# Find probability at these z score
P11 = 0
P12 = 0.2643
# now Beta = p12 - p11
Beta11 = P12 - P11
print(f"Beta at Mu1 = 52 is : {Beta11}")
# Calculate Beta at Mu2 = 50.5
n1 = 10
Sig =2.5
Mu2 =50.5
p13 = (48.5 - Mu2)/(Sig/math.sqrt(n1))
print(p13)
p14 = (51.5 - Mu2)/(Sig/math.sqrt(n1))
print(p14)

# so our z score lies in between these two p13 < z < p14.
# Find probability at these z score
P13 = 0.0057
P14 = 0.8962
# now Beta = p13 +(1 - p14)
Beta12 = P13 +(1- P14)
print(f"Beta at Mu2 = 50.5 is : {Beta12}")

#48 < x < 51
print("#48 < x < 51")
# Calculate Beta at Mu1 = 52
n2 = 10
Sig =2.5
Mu1 =52

p21 = (48.0 - Mu1)/(Sig/math.sqrt(n2))
print(p21)
p22 = (51.0 - Mu1)/(Sig/math.sqrt(n2))
print(p22)

# As p22 > p21, so our z score lies in between these two p22 < z < p21.
# Find probability at these z score
P21 = 0
P22 = 0.1038
# now Beta = p22 - p21
Beta21 = P22 - P21
print(f"Beta at Mu2 = 52 is : {Beta21}")
# Calculate Beta at Mu2 = 50.5
n2 = 10
Sig =2.5
Mu2 =50.5

p23 = (48 - Mu2)/(Sig/math.sqrt(n2))
print(p23)
p24 = (51 - Mu2)/(Sig/math.sqrt(n2))
print(p24)

# so our z score lies in between these two p13 < z < p14.
# Find probability at these z score
P23 = 0.0008
P24 = 0.7357
# now Beta = p13 +(1 - p14)
Beta22 = P23 +(1- P24)
print(f"Beta at Mu2 = 50.5 is : {Beta22}")

#48.81 < x < 51.9
print("#48.81 < x < 51.9")

# Calculate Beta at Mu1 = 52
n3 = 16
Sig =2.5
Mu1 =52

p31 = (48.81 - Mu1)/(Sig/math.sqrt(n3))
print(p31)
p32 = (51.9 - Mu1)/(Sig/math.sqrt(n3))
print(p32)

# so our z score lies in between these two p31 < z < p32.
# Find probability at these z score
P31 = 0.4364
P32 = 0
# now Beta = p31 - p32
Beta31 = P31 - P32
print(f"Beta at Mu2 = 52 is : {Beta31}")

# Calculate Beta at Mu2 = 50.5
n3 = 16
Sig =2.5
Mu2 =50.5

p33 = (48.81 - Mu2)/(Sig/math.sqrt(n3))
print(p33)
p34 = (51.9 - Mu2)/(Sig/math.sqrt(n3))
print(p34)

# so our z score lies in between these two p33 < z < 1- p34.
# Find probability at these z score
P33 = 0.0032
P34 = 0.9875
# now Beta = P33 + 1- P34
Beta32 = P33 +1 - P34
print(f"Beta at Mu2 = 50.5 is : {Beta32}")

#48.42 <x < 51.58
print("#48.42 <x < 51.58")
# Calculate Beta at Mu1 = 52
n4 = 16
Sig =2.5
Mu1 =52

p41 = (48.42 - Mu1)/(Sig/math.sqrt(n4))
print(p41)
p42 = (51.58 - Mu1)/(Sig/math.sqrt(n4))
print(p42)

# As P42> P41, so our z score lies in between these two p42 < z < p41.
# Find probability at these z score
P41 = 0.0
P42 = 0.2514
# now Beta = p31 - p32
Beta41 = P42 - P41
print(f"Beta at Mu2 = 52 is : {Beta41}")

# Calculate Beta at Mu2 = 50.5
n4 = 16
Sig =2.5
Mu2 =50.5

p43 = (48.42 - Mu2)/(Sig/math.sqrt(n4))
print(p33)
p44 = (51.58 - Mu2)/(Sig/math.sqrt(n4))
print(p34)

# so our z score lies in between these two p33 < z < 1- p34.
# Find probability at these z score
P43 = 0.0035
P44 = 0.9875
# now Beta = P43 + 1- P44
Beta42 = P43 +(1 - P44)
print(f"Beta at Mu2 = 50.5 is : {Beta42}")

# Problem statement 8
print("# Problem statement 8")
#t_score = ?
n = 16
p_mean = 10
sample_mean =12
sample_std =1.5
SE = sample_std/(n**0.5)
t = (sample_mean-p_mean)/SE
print(f"t_score is {round((t),1)}")

# Problem statement 9
print("# Problem statement 9")
n= 16
alpha=(1-0.99)/2
print(f"t_score is {stats.t.ppf(1-alpha,df=15)}")

# Problem statement 10
print("# Problem statement 10")
n=25
std=4
mean=60
alpha=(1-0.95)/2
t_score=stats.t.ppf(1-alpha,df=24)
print(f"Range is : {mean+t_score*(std/(n**0.5))} {mean-t_score*(std/(n**0.5))}")
p=stats.t.cdf(0.1,df=24)-stats.t.cdf(-0.05,df=24)
print(f"probability that (−𝑡0.05 <𝑡<𝑡0.10) is {p}")

# Problem statement 11
print("# Problem statement 11")
n1 = 1200
x1 = 452
s1 = 212
n2 = 800
x2 = 523
s2 = 185
s_1=s1**2
s_2=s2**2
alpha=0.05
se=((s_1/n1)+(s_2/n2))**0.5
z_score=(x1-x2)/se
print(f"Z_Score is {z_score}")
print(f"Critical region is {norm.ppf(alpha/2)} {-norm.ppf(alpha/2)}")
print("We reject null hypothesis since it lies within critical region at alpha=5%.So,number of people travelling from Bangalore to Chennai is different from the number of people travelling from Bangalore to Hosur in a week")


# Problem statement 12
print("# Problem statement 12")
n1 = 100
x1 = 308
s1 = 84
n2 = 100
x2 = 254
s2 = 67
s_1=s1**2
s_2=s2**2
alpha=0.05
SE=((s_1/n1)+(s_2/n2))**0.5
z_score=(x1-x2)/SE
print(f"Z_Score is {z_score}")
print(f"Critical region is {norm.ppf(alpha/2)} {-norm.ppf(alpha/2)}")
print("We reject null hypothesis since it lies within critical region at alpha=5%.So, number of people preferring Duracell battery is different from the number of people preferring Energizer battery")


# Problem statement 13
print("# Problem statement 13")
n1 = 14
x1 = 0.317
s1 = 0.12
n2 = 9
x2 = 0.21
s2 = 0.11
s_1=s1**2
s_2=s2**2
s=((n1-1)*s_1)+((n2-1)*s_2)
n=(n1+n2-2)
se=(s/n)**0.5
n_1=((1/n1)+(1/n2))**0.5
t_score=(x1-x2)/se*n_1
print(f"T_score is {t_score}")
print(f"Critical Region is {stats.t.ppf(1-0.05,df=n)}")
print("We accept null hypothesis at alpha=5%.So,average price do not increase")

# Problem statement 14
print("# Problem statement 14")
n1 = 15
x1 = 6598
s1 = 844
n2 = 12
x2 = 6870
s2 = 669
s_1=s1**2
s_2=s2**2
s=((n1-1)*s_1)+((n2-1)*s_2)
n=(n1+n2-2)
se=(s/n)**0.5
n_1=((1/n1)+(1/n2))**0.5
t_score=(x1-x2)/se*n_1
print(f"T_score is {t_score}")
print(f"Critical Region is {stats.t.ppf(0.05,df=n)}")
print("We accept null hypothesis at alpha=5%.So average price remains same")

# Problem statement 15
print("# Problem statement 15")
n1 = 1000
x1 = 53
𝑝1 = 0.53
n2 = 100
x2 = 43
𝑝2= 0.53
p=(x1+x2)/(n1+n2)
n=(1/n1)+(1/n2)
p_1=p*(1-p)
Z=(p1-p2)/((p_1*n)**0.5)
print(f"Z_score is {Z}")
print(f"Critical region is {norm.ppf(0.05)}")
print("We can't reject null hypothesis at alpha=10%")

# Problem statement 16
print('# Problem statement 16')
n1 = 300
x1 = 120
p1 = 0.40
n2 = 700
x2 = 140
p2= 0.20
p=(x1+x2)/(n1+n2)
n=(1/n1)+(1/n2)
p_1=p*(1-p)
Z=(p1-p2-0.1)/((p_1*n)**0.5)
print(f"Z_score is {Z}")
print(f"Critical region is {-norm.ppf(0.05)}")
print("We reject null hypothesis at alpha=5%")


# Problem statement 17
print('# Problem statement 17')
f_obs= [16, 20, 25, 14, 29, 28]
f_exp= [22,22,22,22,22,22]
result=stats.chisquare(f_obs,f_exp)
print(f"Chi square value is {result[0]} and p-value is {result[1]}")
print('Dias is unbiased.')

# Problem statement 18
print("# Problem statement 18")
observed_voted_men=2792
observed_voted_women=3591
observed_not_voted_men=1486
observed_not_voted_women=2131
total_voted=2792+3591
total_not_voted=1486+2131
total_men=2792+1486
total_women=3591+2131
expected_voted_men=(total_voted*total_men)/10000
expected_voted_women=(total_voted*total_women)/10000
expected_not_voted_men=(total_not_voted*total_men)/10000
expected_not_voted_women=(total_not_voted*total_women)/10000
chisquare1=(((observed_voted_women-expected_voted_women)**2)/expected_voted_women)
chisquare2=(((observed_voted_men-expected_voted_men)**2)/expected_voted_men)
chisquare3=(((observed_not_voted_men-expected_not_voted_men)**2)/expected_not_voted_men)
chisquare4=(((observed_not_voted_women-expected_not_voted_women)**2)/expected_not_voted_women)
chisquare=chisquare1+chisquare2+chisquare3+chisquare4
print(f"Chi Square value is {chisquare}")
print(f"Critical region with alpha=0.05 is 3.84")
print("We reject null hypothesis.It is not gender and voting independent")

# Problem statement 19
print("# Problem statement 19")
obs=[41,19,24,16]
exp=[25,25,25,25]
result=stats.chisquare(obs,exp)
print(f"Chi Square value is {result[0]}")
print(f"Critical region with 3df and alpha=0.05 is 7.82")
print("We reject null hypothesis. All candidates are not equally popular")

# Problem statement 20
print("# Problem statement 20")
obs=([[18,22,20],[2,28,40],[20,10,40]])
result=chi2_contingency(obs)
print(f"Chi Square value is {result[0]}")
print(f"Critical region with 4df and alpha=0.001 is 18.47")
print("We reject null hypothesis.There is significant relationship between age and photograph preference")

# Problem statement 21
print("# Problem statement 21")
obs=np.array([[18,40],[32,10]])
result=chi2_contingency(obs)
print(f"Chi Square value is {result[0]}")
print(f"Critical region with 1df and alpha=0.001 is 10.83")
print("We reject null hypoythesis.So,there is significant difference between the 'support' and 'no support' conditions in the frequency with which individuals are likely to conform")


# Problem statement 22
print("# Problem statement 22")
obs=([[12,32],[22,14],[9,6]])
result=chi2_contingency(obs)
print(f"Chi Square value is {result[0]}")
print(f"Critical region with 2df and alpha=0.001 is 13.82")
print("We accept null hypothesis.there is no relationship between height and leadership qualities")

# Problem statement 23
print("# Problem statement 23")
obs = np.array([[679,103,114], [63,10,20],[42,18,25]])
result=chi2_contingency(obs)
print(f"Chi Square value is {result[0]}")
print(f"Critical region with 4df and alpha=0.001 is 18.47")
print("We reject null hypothesis at alpha=0.001 .there is relationship between martial status and employment status")