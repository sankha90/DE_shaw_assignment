# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:19:04 2020

@author: subhra

"""

#importing neccessary packages
import pandas as pd
import os
from PIL import Image
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import altair as alt
import streamlit as st

DATA_URL = ("c://Users//subhr//final_data.csv")

#reading the data
@st.cache(allow_output_mutation=True)
def load_data():
	df = pd.read_csv("c://Users//subhr//final_data.csv")
	return df

df = load_data()

#extracting day and year from the date
df['Day'] = pd.DatetimeIndex(df['Conf_Date']).day_name()
df['Conf_Year'] = pd.DatetimeIndex(df['Conf_Date']).year
df['Application_Year'] = pd.DatetimeIndex(df['Application_date']).year

st.markdown("<h1 style='text-align: center; color: BROWN;'>Recrutiing Activity ' 2018 </h1>", unsafe_allow_html=True)


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

st.warning("There are 4959 columns and 11 columns , For 3 candidates dates were wrongly mentioned , it was corrected before the analysis. For Experienced candidates the **number of candidates who had in-house interviews** is **>** **no. of candidates for initial phone screening** which is not possible in real life hence adjustments have been made accordingly")

st.markdown("For the analysis I have considered **12** KPI's , they are:")
st.markdown("**1.** Acceptance rate in 2018 by **Department**")
st.markdown("**2.** How are the campus candidates progressing through the recruiiting process , any notable      differences across certain type of candidates")
st.markdown("**3.** Recruitment rate by **Position**")
st.markdown("**4.** Qualified candidates per hire by **Department**")
st.markdown("**5.** Time to hire by **Department**")
st.markdown("**6.** Offer acceptance rate")
st.markdown("**7.** Yield ratio")
st.markdown("**8.** Sourcing channel effectiveness")
st.markdown("**9.** Candidate diversity")
st.markdown("**10.** Mean experience by **Department**")
st.markdown("**11.** Offer letter acceptance rate by day of the week")
st.markdown("**12.** Changes and potential areas to focus to develope a cloud architechture across the organization")


st.markdown("<h1 style='text-align: left; color: BROWN;'>1. Acceptance rate in 2018 by Department</h1>", unsafe_allow_html=True)

#filtering 2018 data
acc = df.loc[df['Application_Year'].isin([2018])] 
acc = acc.reset_index(drop=False)


#aggregating by Department
#af2_1=acc.loc[acc['Event'].isin([0 , 1 , 2])]
a3 = acc.groupby(['Department'])['ID'].count()

#Seperating those candidates who have accepted and aggregating by different Pos
af2=acc[(acc.Event == 1) & (acc.Conf_Year == 2018)]
a4 = af2.groupby('Department')['ID'].count()

#applications per hire
applications_per_hire_dep_18 = pd.merge(a3,a4,how = 'outer', on = 'Department')
applications_per_hire_dep_18.rename(columns = {'ID_x':'Count', 'ID_y':'Hired'}, inplace = True)
applications_per_hire_dep_18['Ratio'] = round((applications_per_hire_dep_18['Hired']/applications_per_hire_dep_18['Count'])*100,2)
applications_per_hire_dep_18 = applications_per_hire_dep_18.reset_index(drop=False)
applications_per_hire_dep_18 = applications_per_hire_dep_18.sort_values(["Ratio"], ascending = (False))

sns.lmplot( x="Department", y="Ratio", data=applications_per_hire_dep_18, fit_reg=False, hue='Department', legend=True, palette="deep").fig.suptitle("Hiring by Department in 2018")

if st.checkbox('Show hiring trend in 2018'):
    st.subheader('Hiring trend in 2018 by Department')
    st.dataframe(applications_per_hire_dep_18.style.highlight_max(axis=0).format({'Ratio':'{:.1f}','Hired':'{:.0f}'}))


if st.checkbox('Show hiring trend in 2018 plot'):
    st.subheader('Hiring trend in 2018 by Department in plot')
    st.pyplot()


st.markdown("<h1 style='text-align: left; color: BROWN;'>2. Campus candidate progress & Observation</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left; color: BROWN;'>Campus hiring progress & Observation</h3>", unsafe_allow_html=True)


#filtering only the campus candidates
campus = df.loc[df['Type'] == 'Campus'] 
#there are a total of 4131 candidates from all category of offer
#count of candidates applied by year
c1 = campus.groupby('Application_Year')['ID'].nunique()
c1 = c1.reset_index(drop=False)
alt.Chart(c1).mark_bar().encode(
    x='Application_Year',
    y='ID'
)

#Now we will dig a little deeper and see how campus candidates are performing through the years
campus_16 = campus.loc[df['Application_Year'] == 2016] 
campus_17 = campus.loc[df['Application_Year'] == 2017] 
campus_18 = campus.loc[df['Application_Year'] == 2018] 


#overall campus hire trend
#Pass through rate
c8 = campus.groupby('Stage')['ID'].nunique()
c8 = c8.reset_index(drop=False)
c8.set_index('Stage',inplace = True)
#adding 93 candidates who have already crossed this phases
c8.loc['New Application','ID'] = c8.loc['New Application','ID']+93
c8.loc['Phone Screen','ID'] = c8.loc['Phone Screen','ID']+93
c8.loc['In-House Interview','ID'] = c8.loc['In-House Interview','ID']+93
c8 = c8.sort_values(["ID"], ascending = (False))
c8 = c8.reset_index(drop=False)
c_row = pd.Series(['Accepted', 49])
c_row_df = pd.DataFrame([c_row])
c_row_df.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
c8 = pd.concat([c8, c_row_df], ignore_index=True)
c8.set_index('Stage',inplace = True)
c8['Percentage'] =(c8[['ID']].pct_change()[:6])
c8['Percentage'] = abs(c8['Percentage'])
c8['Percentage'] =np.round((1-c8.loc[:,'Percentage'].values)*100)
c8 = c8.reset_index(drop=False)


if st.checkbox('Show campus hiring trend overall'):
    st.subheader('Campus hiring trend overall data')
    st.dataframe(c8.style.format({'Percentage':'{:.2f}'}).highlight_max(axis=0))
    st.write('The pass through rate shows the total number of new applicants were **2942** , from that only **25%** went to the next stage ~ (Phone screen).After phone screen **87%** gets a call for the inhouse interview and from that group of **638** candidates only **15%** got an offer letter which is **93** Candidates and out of **93** candidates **49** candidates have accepted it , i.e **53%**.')


#pass through rate in 2016
c16 = campus_16.groupby('Stage')['ID'].nunique()
c16 = c16.reset_index(drop=False)
c16.set_index('Stage',inplace = True)
#adding 93 candidates who have already crossed this phases
c16.loc['New Application','ID'] = c16.loc['New Application','ID']+28
c16.loc['Phone Screen','ID'] = c16.loc['Phone Screen','ID']+28
c16.loc['In-House Interview','ID'] = c16.loc['In-House Interview','ID']+28
c16 = c16.sort_values(["ID"], ascending = (False))
c16 = c16.reset_index(drop=False)
c_row_16 = pd.Series(['Accepted', 15])
c_row_df_16 = pd.DataFrame([c_row_16])
c_row_df_16.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
c16 = pd.concat([c16, c_row_df_16], ignore_index=True)
c16.set_index('Stage',inplace = True)
c16['Percentage'] =(c16[['ID']].pct_change()[:6])
c16['Percentage'] = abs(c16['Percentage'])
c16['Percentage'] =np.round((1-c16.loc[:,'Percentage'].values)*100)
c16 = c16.reset_index(drop=False)
c16['Year'] = 2016



#pass through rate in 2017
c17 = campus_17.groupby('Stage')['ID'].nunique()
c17 = c17.reset_index(drop=False)
c17.set_index('Stage',inplace = True)
#adding 93 candidates who have already crossed this phases
c17.loc['New Application','ID'] = c17.loc['New Application','ID']+23
c17.loc['Phone Screen','ID'] = c17.loc['Phone Screen','ID']+23
c17.loc['In-House Interview','ID'] = c17.loc['In-House Interview','ID']+23
c17 = c17.sort_values(["ID"], ascending = (False))
c17 = c17.reset_index(drop=False)
c_row_17 = pd.Series(['Accepted', 17])
c_row_df_17 = pd.DataFrame([c_row_17])
c_row_df_17.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
c17 = pd.concat([c17, c_row_df_17], ignore_index=True)
c17.set_index('Stage',inplace = True)
c17['Percentage'] =(c17[['ID']].pct_change()[:6])
c17['Percentage'] = abs(c17['Percentage'])
c17['Percentage'] =np.round((1-c17.loc[:,'Percentage'].values)*100)
c17 = c17.reset_index(drop=False)
c17['Year'] = 2017


#pass through rate in 2017
c18 = campus_18.groupby('Stage')['ID'].nunique()
c18 = c18.reset_index(drop=False)
c18.set_index('Stage',inplace = True)
#adding 93 candidates who have already crossed this phases
c18.loc['New Application','ID'] = c18.loc['New Application','ID']+41
c18.loc['Phone Screen','ID'] = c18.loc['Phone Screen','ID']+41
c18.loc['In-House Interview','ID'] = c18.loc['In-House Interview','ID']+41
c18 = c18.sort_values(["ID"], ascending = (False))
c18 = c18.reset_index(drop=False)
c_row_18 = pd.Series(['Accepted', 18])
c_row_df_18 = pd.DataFrame([c_row_18])
c_row_df_18.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
c18 = pd.concat([c18, c_row_df_18], ignore_index=True)
c18.set_index('Stage',inplace = True)
c18['Percentage'] =(c18[['ID']].pct_change()[:6])
c18['Percentage'] = abs(c18['Percentage'])
c18['Percentage'] =np.round((1-c18.loc[:,'Percentage'].values)*100)
c18 = c18.reset_index(drop=False)
c18['Year'] = 2018

f2 = pd.concat([c16, c17, c18], ignore_index=True)
f2['Type'] = 'Student'
f2 = f2.reset_index(drop=False)


if st.checkbox('Show campus hiring trend in 2016,2017,2018 in one table'):
    st.subheader('Campus hiring trend in 2016,2017,2018')
    st.dataframe(f2.style.format({'Percentage':'{:.2f}'}))
    
Yr = st.multiselect('Select an  year to see yearly trend', f2['Year'].unique())
new_df = f2[(f2['Year'].isin(Yr))]
st.dataframe(new_df)
st.write('Although the number of new candidates are much higher in 2018 the acceptance rate for campus students is the highest in 2017 , **74%** then in 2016 **54%** and then 2018 **44%**. The overall passthrough rate from different stages has remained same over these years but the acceptance rate has gone down and it demands a second look.')



st.markdown("<h3 style='text-align: left; color: BROWN;'>Experienced candidates recruitment progress & Observation</h3>", unsafe_allow_html=True)

#===================================================================================================
#filtering only experienced candidates
exp = df.loc[df['Type'] == 'Experienced'] 
#there are a total of 4131 candidates from all category of offer
#count of candidates applied by year
e1 = exp.groupby('Application_Year')['ID'].nunique()
e1 = e1.reset_index(drop=False)
alt.Chart(e1).mark_bar().encode(
    x='Application_Year',
    y='ID'
)

#Now we will dig a little deeper and see how campus candidates are performing through the years
exp_16 = exp.loc[df['Application_Year'] == 2016] 
exp_17 = exp.loc[df['Application_Year'] == 2017] 
exp_18 = exp.loc[df['Application_Year'] == 2018] 


#overall Lateral hire trend
#Pass through rate
e8 = exp.groupby('Stage')['ID'].nunique()
e8 = e8.reset_index(drop=False)
e8.set_index('Stage',inplace = True)
#adding 31 candidates who have already crossed this phases
e8.loc['New Application','ID'] = e8.loc['New Application','ID']+31
e8.loc['Phone Screen','ID'] = e8.loc['Phone Screen','ID']+31
e8.loc['In-House Interview','ID'] = e8.loc['In-House Interview','ID']+31
e8 = e8.sort_values(["ID"], ascending = (False))
e8 = e8.reset_index(drop=False)
e_row = pd.Series(['Accepted', 19])
e_row_df = pd.DataFrame([e_row])
e_row_df.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
e8 = pd.concat([e8, e_row_df], ignore_index=True)
e8.set_index('Stage',inplace = True)
e8['Percentage'] =(e8[['ID']].pct_change()[:6])
e8['Percentage'] = abs(e8['Percentage'])
e8['Percentage'] =np.round((1-e8.loc[:,'Percentage'].values)*100,2)
e8 = e8.reset_index(drop=False)

if st.checkbox('Show lateral hiring trend overall'):
    st.subheader('lateral hiring trend overall data')
    st.dataframe(e8.style.highlight_max(axis=0).format({'Percentage':'{:.1f}'}))
    st.write('The pass through rate shows the total number of new applicants were **546** , from that **35%** went to the next stage ~ (Phone screen).After phone screen **77%** were selcted for the inhouse interview and from that group of **150** candidates **20%** got an offer letter which is **31** Candidates and out of **31** candidates **19** candidates have accepted it , i.e **61%**.')



#pass through rate in 2016
e16 = exp_16.groupby('Stage')['ID'].nunique()
e16 = e16.reset_index(drop=False)
e16.set_index('Stage',inplace = True)
#adding 7 candidates who have already crossed this phases
e16.loc['New Application','ID'] = e16.loc['New Application','ID']+7
e16.loc['Phone Screen','ID'] = e16.loc['Phone Screen','ID']+7
e16.loc['In-House Interview','ID'] = e16.loc['In-House Interview','ID']+7
e16 = e16.sort_values(["ID"], ascending = (False))
e16 = e16.reset_index(drop=False)
e_row_16 = pd.Series(['Accepted', 4])
e_row_df_16 = pd.DataFrame([e_row_16])
e_row_df_16.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
e16 = pd.concat([e16, e_row_df_16], ignore_index=True)
e16.set_index('Stage',inplace = True)
e16['Percentage'] =(e16[['ID']].pct_change()[:6])
e16['Percentage'] = abs(e16['Percentage'])
e16['Percentage'] =np.round((1-e16.loc[:,'Percentage'].values)*100)
e16 = e16.reset_index(drop=False)
e16['Year'] = 2016


#pass through rate in 2017
e17 = exp_17.groupby('Stage')['ID'].nunique()
e17 = e17.reset_index(drop=False)
e17.set_index('Stage',inplace = True)
#adding 11 candidates who have already crossed this phases
e17.loc['New Application','ID'] = e17.loc['New Application','ID']+11
e17.loc['Phone Screen','ID'] = e17.loc['Phone Screen','ID']+11
e17.loc['In-House Interview','ID'] = e17.loc['In-House Interview','ID']+11
e17 = e17.sort_values(["ID"], ascending = (False))
e17 = e17.reset_index(drop=False)
e_row_17 = pd.Series(['Accepted', 9])
e_row_df_17 = pd.DataFrame([e_row_17])
e_row_df_17.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
e17 = pd.concat([e17, e_row_df_17], ignore_index=True)
e17.set_index('Stage',inplace = True)
e17['Percentage'] =(e17[['ID']].pct_change()[:6])
e17['Percentage'] = abs(e17['Percentage'])
e17['Percentage'] =np.round((1-e17.loc[:,'Percentage'].values)*100)
e17 = e17.reset_index(drop=False)
e17['Year'] = 2017


#pass through rate in 2018
e18 = exp_18.groupby('Stage')['ID'].nunique()
e18 = e18.reset_index(drop=False)
e18.set_index('Stage',inplace = True)
#adding 13 candidates who have already crossed this phases
e18.loc['New Application','ID'] = e18.loc['New Application','ID']+13
e18.loc['Phone Screen','ID'] = e18.loc['Phone Screen','ID']+13
e18.loc['In-House Interview','ID'] = e18.loc['In-House Interview','ID']+13
e18 = e18.sort_values(["ID"], ascending = (False))
e18 = e18.reset_index(drop=False)
e_row_18 = pd.Series(['Accepted', 5])
e_row_df_18 = pd.DataFrame([e_row_18])
e_row_df_18.rename(columns = {0:'Stage', 1:'ID'}, inplace = True)
e18 = pd.concat([e18, e_row_df_18], ignore_index=True)
e18.set_index('Stage',inplace = True)
e18['Percentage'] =(e18[['ID']].pct_change()[:6])
e18['Percentage'] = abs(e18['Percentage'])
e18['Percentage'] =np.round((1-e18.loc[:,'Percentage'].values)*100)
e18 = e18.reset_index(drop=False)
e18['Year'] = 2018


#f = pd.concat([e16, e17, e18], axis=1, join='outer')
f1 = pd.concat([e16, e17, e18], ignore_index=True)
f1['Type'] = 'Experienced'


if st.checkbox('Show lateral hiring trend in 2016,2017,2018 in one table'):
    st.subheader('lateral hiring trend in 2016,2017,2018')
    st.dataframe(f1)

yr_1 = st.slider('Select a year by dragging the slider accordingly', 2016, 2018, 2017)
new_df_1 = f1[f1.Year.isin([yr_1])]
st.dataframe(new_df_1)
st.write('Although the number of new candidates are much higher in 2018 the acceptance rate for campus students is the highest in 2017 , **82%** then in 2016 **57%** and then 2018 **38%**. The overall pass through rate from different stages has remained same over these years but the acceptance rate has gone down and it demands a second look.')


st.markdown("<h3 style='text-align: left; color: BROWN;'>Observation</h3>", unsafe_allow_html=True)
st.write('Freshers and experienced candidates both have an acceptance rate of **53%** and **61%** respectively and we have seen an increase in acceptance rate for the year **2017**. The pass through rate sharply decreases in 2018 which needs some attention.')

#======================================================================================
# Position wise hiring percentage
#======================================================================================
#aggregating by Pos
st.markdown("<h1 style='text-align: left; color: BROWN;'>3. Recruitment rate by Position</h1>", unsafe_allow_html=True)

g9=df.loc[df['Event'].isin([0 , 1 , 2])]
g1 = g9.groupby('Pos')['ID'].nunique()


#Seperating those candidates who have accepted and aggregating by different Pos
df2=df.loc[df['Event'] == 1]
g2 = df2.groupby('Pos')['ID'].nunique()

#applications per hire
applications_per_hire = pd.merge(g1,g2,how = 'outer', on = 'Pos')
applications_per_hire.rename(columns = {'ID_x':'Application_count', 'ID_y':'Hired'}, inplace = True)
applications_per_hire['Ratio'] = round((applications_per_hire['Hired']/applications_per_hire['Application_count'])*100)
applications_per_hire = applications_per_hire.reset_index(drop=False)
applications_per_hire = applications_per_hire.sort_values(["Ratio"], ascending = (False))


if st.checkbox('Hiring trend by Position'):
    st.subheader('Overall hiring trend by Position')
    st.dataframe(applications_per_hire.style.format({'Hired':'{:.1f}','Ratio':'{:.2f}'}).highlight_max(axis=0))
    st.write('From the table we can see Financial analyst , IT , Account Ex have a high number of accepted candidates however this does not align with the current vision of further developing the cloud - based software suite')



#======================================================================================
# Qualified candidates by department
#======================================================================================

st.markdown("<h1 style='text-align: left; color: BROWN;'>4. Qualified candidates per hire by Department</h1>", unsafe_allow_html=True)

#qualified candidates who have completed round_1
g5=df.loc[df['Stage_class'].isin([1 , 2 , 3])]
g5 = g5.groupby('Department')['ID'].nunique()

#candidates who have accepted and aggregating by different department
df2=df.loc[df['Event'] == 1]
g6 = df2.groupby('Department')['ID'].nunique()
#applications per hire
qual_cand_per_hire_dep = pd.merge(g5,g6,how = 'outer', on = 'Department')
qual_cand_per_hire_dep.rename(columns = {'ID_x':'qual_cand_count', 'ID_y':'Hired'}, inplace = True)
qual_cand_per_hire_dep['Ratio'] = round((qual_cand_per_hire_dep['Hired']/qual_cand_per_hire_dep['qual_cand_count'])*100,2)
qual_cand_per_hire_dep = qual_cand_per_hire_dep.reset_index(drop=False)
qual_cand_per_hire_dep = qual_cand_per_hire_dep.sort_values(["Ratio"], ascending = (False))

if st.checkbox('Qualified candidates per hire by Department'):
    st.subheader('Qualified candidates per hire by Department')
    st.dataframe(qual_cand_per_hire_dep.style.format({'Hired':'{:.1f}','Ratio':'{:.1f}'}).highlight_max(axis=0))
    st.write('IT has the most amount of qualified candidaes (*who have successfully competed initial rounds and acceppted offers*) compared to other departments')


sns.lmplot( x="Department", y="Ratio", data=qual_cand_per_hire_dep, fit_reg=False, hue='Department', legend=True, palette="deep").fig.suptitle("Qualified applicants per hire by Department")

if st.checkbox('Qualified candidates / hire by Department plot'):
    st.subheader('Qualified candidates / hire by Department in plot')
    st.pyplot()


#======================================================================================
# Time to hire by depertment
#======================================================================================

st.markdown("<h1 style='text-align: left; color: BROWN;'>5. Time to hire by Department</h1>", unsafe_allow_html=True)

#Time to hire
df2['Recency'] = df2.Recency.astype(int)
g7 = df2.groupby('Department')['Recency'].mean()
Time_to_hire = g7.reset_index(drop=False)
Time_to_hire = Time_to_hire.sort_values(["Recency"], ascending = (True))

if st.checkbox('Time to hire hire by Department'):
    st.subheader('Time to hire by Department Plot')
    st.dataframe(Time_to_hire.style.format({'Recency':'{:.1f}'}).highlight_max(axis=0))
    st.write('Engineering depertment takes **41 days** to recuit a candidate , followed by Finance and IT')


#======================================================================================
# Offer acceptance rate
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>6. Offer acceptance Rate</h1>", unsafe_allow_html=True)


#Offer acceptance rate
g9=df.loc[df['Event'].isin([0 , 1 , 2])]
g10 = g9.groupby(['Offer_taken'])['ID'].nunique()
g10 = g10.reset_index(drop=False)
g10['PCT'] = round((g10['ID']/g10['ID'].sum())*100)

g11 = g9.groupby(['Offer_taken','Type'])['ID'].nunique()
g11 = g11.reset_index(drop=False)
g11['PCT'] = round((g11['ID']/g11['ID'].sum())*100,2)
g11_tr = g11.transpose() #offer acceptance , rejection and pending by Candidate type
if st.checkbox('Offer status rate by candidate type'):
    st.dataframe(g11.style.highlight_max(axis=0).format({'PCT':'{:.1f}'}))
    st.write('Campus students have the highest amount of acceptance rate')



g12 = g9.groupby(['Offer_taken','Department'])['ID'].nunique()
g12 = g12.reset_index(drop=False)
g12['PCT'] = round((g12['ID']/g12['ID'].sum())*100,2)
if st.checkbox('Offer status rate by Department'):
    st.dataframe(g12.style.highlight_max(axis=0).format({'PCT':'{:.1f}'}))
    st.write('Engineering department has the highest amount of acceptance rate')


#======================================================================================
# Yield Ratio
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>7. Yield Ratio</h1>", unsafe_allow_html=True)

#Yield ratio
Yield = round((g9['ID'].count()/df['ID'].count()*100),2)
if st.checkbox('Show Yield Ratio'):
    st.write('**Yield** = # of applicants that were interviewed or have cleared the initial screening / total number of applications generated')
    st.write(Yield , 'Current Yield ratio is really low compared to industry standard')



#======================================================================================
# Sourcing channel effecctiveness
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>8. Sourcing channel effectiveness</h1>", unsafe_allow_html=True)


#Sourcing channel effectiveness , total 124 who have responded

s10 = g9.groupby(['Offer_taken','App_source'])['ID'].nunique()
s10 = s10.reset_index(drop=False)
s10['PCT'] = round((s10['ID']/s10['ID'].sum())*100,2)
s10_tr = s10.transpose()

if st.checkbox('Sourching channel effectiveness for candidates with an outcome'):
    st.dataframe(s10.style.highlight_max(axis=0).format({'PCT':'{:.1f}'}))
    st.write('Campus events have the highest amount of visibility , a key area to focus on. This table has 124 candidats with an outcome.')


#source of hire #total 4959 from where candidates are coming
s11 = df.groupby(['App_source','Stage'])['ID'].nunique()
s11 = s11.reset_index(drop=False)
s11['PCT'] = round((s11['ID']/s11['ID'].sum())*100,2) 
if st.checkbox('Overall Sourching channel effectiveness by respective stages'):
    st.dataframe(s11.style.highlight_max(axis=0))
    st.write('In terms of overall effectiveness **Campus job boards** and **Campus job events** have the most amount of visibility. **25** of the total new applications come from job boards.We can see Agencies are not contributing as much as organic sources.')



#======================================================================================
# Hiriing Diversity
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>9. Hiring Diversity</h1>", unsafe_allow_html=True)


b = st.checkbox("Show hiring diversity")
if b:
    Image_1 = Image.open('C://Users//subhr//Desktop/Diversity.png')
    st.image(Image_1,width=500)
    st.write('Candidates have a good mixture of different backgrounds , i.e : Bachelors , Masters and Phd.')


#======================================================================================
# Experience by Team
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>10. Average experience </h1>", unsafe_allow_html=True)


e11 = df.groupby(['Department'])['Experience'].mean()
e11 = e11.reset_index(drop=False)
e11 = e11.sort_values(["Experience"], ascending = (False))

if st.checkbox('Average experience by Department'):
    st.dataframe(e11.style.highlight_max(axis=0))
    st.write('Operations team have the most number of experienced candidates **8%** followed by IT team **7.4%**.')


#======================================================================================
# Best day to send or not send an offer letter
#======================================================================================
st.markdown("<h1 style='text-align: left; color: BROWN;'>11. When to send / hold Offer Letter </h1>", unsafe_allow_html=True)

a11 = df2.groupby('Day')['ID'].nunique()
a11 = a11.reset_index(drop=False)
a11 = a11.sort_values(["ID"], ascending = (False))
a11['PCT'] = round((a11['ID']/a11['ID'].sum())*100,2)

if st.checkbox('Show when to send Offer letter'):
    st.dataframe(a11.style.highlight_max(axis=0).format({'PCT':'{:.1f}'}))



#best day to not send an offer letter
df3=df.loc[df['Event'] == 0]
n11 = df3.groupby('Day')['ID'].nunique()
n11 = n11.reset_index(drop=False)
n11 = n11.sort_values(["ID"], ascending = (False))
n11['PCT'] = round((n11['ID']/n11['ID'].sum())*100,2)
if st.checkbox('Show when to hold Offer letter'):
    st.dataframe(n11.style.highlight_max(axis=0).format({'PCT':'{:.1f}'}))
    st.write('On Mondays **19%** offer got accepted and on Fridays **22%** offers got rejected.')