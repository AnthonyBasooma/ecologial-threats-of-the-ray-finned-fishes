# ==============================================================================
# Extiction rate of the freshwater, marine and selected mammals
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import os
from glob import glob
import scipy.stats as st
import geopandas as gpd
import re
#===============================================================================
#==============================================================================

os.chdir('E:\\ray finned\\manuscript\\extinction\\data analysis')

#1. All species data from GBIF
#==============================================================================
spdata = pd.read_csv('exdata.csv', engine='pyarrow')
redata = pd.read_csv('regdata.csv', engine='pyarrow')
gsdata = pd.read_csv('globalsp.csv', engine='pyarrow')
thdata = pd.read_csv('threatfiles.csv', engine='python')
actdata = pd.read_csv('actinodata.csv', engine='python').filter(items=['species', 'iucnall', 'systems'])
cdata = pd.read_csv('cdata2.csv', engine='python', encoding='cp1252')
spdata2 = pd.read_csv('spcontclean.csv', engine='python', encoding='cp1252', skipinitialspace=True)
#==============================================================================


#family summaries


fmspp = spdata.groupby(['continent','family','species'], as_index=False).\
    agg(ct = ('species', len)).replace(r'^\s*$', np.nan, regex=True).\
    dropna(subset = ['species', 'family'], inplace=False).assign(ct = 1).\
    groupby(['species','family'], as_index=False).\
    agg(cts = ('ct', sum))

spconts = spdata2.groupby(['continent','species'], as_index=False).\
    agg(ct = ('species', len)).merge(fmspp, how='left', on='species').assign(ct =1).\
    groupby(['continent','family'], as_index=False).\
    agg(cts = ('ct', len))

spconts.to_csv('famcont2.csv', index=False)


famil = spdata.groupby(['family', 'species'], as_index=False).agg(ft = ('family', len)).\
    replace(r'^\s*$', np.nan, regex=True).dropna(subset=['species'], inplace=False).assign(ct= 1).\
    groupby(['family'], as_index=False, dropna=False).agg(ft2 = ('ct', len)).\
    sort_values('ft2', ascending=False).query('family.notnull()').\
    assign(addd = 'ads', pct = famil.ft2/famil.groupby('addd')['ft2'].transform(sum)*100).\
    query('ft2<10')



#=========
spdn = spdata2.groupby(['continent', 'countryCode'], as_index=False, dropna=False).\
    agg(ctl = ('continent', len)).assign(ct = 1).\
    groupby(['countryCode'], as_index=False, dropna=False).\
    agg(ctl = ('countryCode', len)).query('ctl>1')


species = spdata.filter(items= ['iucnRedListCategory', 'species']).\
    replace(r'^\s*$', "NE", regex=True).\
    groupby(['species','iucnRedListCategory'], as_index= False, dropna=False).\
    agg(ct = ('iucnRedListCategory', len)).assign(ct2 =1).\
    groupby(['species', 'iucnRedListCategory'], as_index=False, dropna=False).agg(tot = ('ct2', sum))

spg = species.assign(ctt= species.groupby('species', as_index=False)['tot'].transform('sum')).\
    query('ctt>1 & iucnRedListCategory !="NE"')

spg2= spg.assign(ctt2= spg.groupby('species', as_index=False)['tot'].transform('sum')).\
    query('ctt2>1')
#
##=========================================================================================
#1. assesing ecological threats affecting actinopterygii
#=============
#summary of threats
#============================================
thlist2 = thdata.groupby('classify', as_index=False).agg(ttot = ('classify', len)).classify.tolist()

sydata = thdata.groupby(['species', 'classify2'], as_index=False).agg(syn = ('species', len)).\
    assign(ct =1).filter(items=['species', 'classify2', 'ct']).\
    pivot_table(index= 'species', columns= 'classify2', values='ct').\
    reset_index().\
    assign(overexplo2= np.where(sydata.overexplo == 1, 'overexploitation', 'not'),
           pollution2 = np.where(sydata.pollution == 1, 'pollution','not'),
           naturalcalamaities2 = np.where(sydata.naturalcalamaities == 1, 'natural calamities', 'not'),
           climatechange2 = np.where(sydata.climatechange == 1, 'climate change', 'not'),
           habitaloss2 = np.where(sydata.habitaloss == 1, 'habitat loss', 'not'),
           nonnativesspp2 = np.where(sydata.nonnativesspp == 1, 'non-native species', 'not'),
           waterabs2=np.where(sydata.waterabs == 1, 'water abstraction', 'not'),
           thfin = sydata.overexplo2+'_'+sydata.pollution2+'_'+sydata.naturalcalamaities2+'_'+
                   sydata.climatechange2+'_'+sydata.habitaloss2+'_'+sydata.nonnativesspp2+'_'+sydata.waterabs2,
           tlookup=sydata.thfin.str.findall('|'.join(thlist2), flags=re.IGNORECASE).apply(set).str.join('_'))

thfinal = sydata.groupby('tlookup', as_index=False, ).agg(freq = ('tlookup', len))
#sydata.to_csv('synergydata.csv', index=False)

#append(sydata.sum(numeric_only=True).rename('total'), ignore_index=True).replace(np.nan, 'total')

speffects = thdata.groupby(['species', 'classify2'], as_index=False).agg(syn = ('species', len)).\
    assign(ct = 1, cts = speffects.groupby(['species'])['ct'].transform(sum)).query('cts>5')


#species with multiple stressors
ydata = thdata.groupby(['species', 'classify'], as_index=False).agg(syn = ('species', len)).\
    assign(ct =1, tot = ydata.groupby('species')['ct'].transform(sum)).\
    filter(items=['species', 'tot']).drop_duplicates(subset=['species'], keep='first').\
    groupby('tot', as_index=False).agg(freq = ('tot', len))

plt.rcParams['font.family'] = 'Cambria'
g = plt.bar(ydata['tot'], height= ydata['freq'])
plt.ylabel('Number of species')
plt.xlabel('Number ecological stressors')

#savefig("stress.png", dpi=400, bbox_inches='tight')

tdataf = thdata.assign(ct = 1).\
    groupby(['species', 'classify', 'iucn', 'systems', 'popnTrend'], as_index=False).\
    agg(tot = ('ct', sum))

spcont = spdata2.groupby(['continent','countryCode','species'], as_index=False, dropna=False).\
    agg(spct = ('species', len)).filter(items=['continent', 'species']).\
    merge(tdataf, how="left", on="species").\
    dropna(subset=['species', 'classify'], inplace=False).assign(ct =1, threats = 'threat')

#===================

overal = spcont.groupby(['classify', 'threats'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).\
    assign(pct = overal.spt/overal.groupby('threats')['spt'].transform('sum')*100)

#donut plot
plt.rcParams['font.family'] = 'Cambria'
my_circle = plt.Circle((0, 0), 0.5, color='white')

# Give color names
plt.pie(overal.pct, labels=overal.classify, autopct='%1.1f%%', pctdistance=0.8,
        labeldistance= 1,
        colors=['#7F00FF', '#E31B23', '#005CAB', '#DCEEF3', '#FFC325', '#007FFF', 'cyan'])

p = plt.gcf()
p.gca().add_artist(my_circle)

#region specific threats
#=====
thrext = spcont.groupby(['continent','iucn', 'classify'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).\
    assign(pct = thrext.spt/thrext.groupby(['iucn', 'continent'])['spt'].transform('sum')*100).\
    filter(items=['continent', 'iucn', 'spt', 'pct', 'classify']).\
    pivot_table(index= ['continent', 'iucn'], columns= 'classify', values='pct').\
    reset_index()

hdata = spcont.groupby(['continent', 'species','iucn', 'classify', 'systems'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).assign(ct =1).\
    groupby(['continent', 'iucn', 'classify', 'systems'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).\
    assign(pct = hdata.spt/hdata.groupby(['continent','iucn', 'systems'])['spt'].transform('sum')*100)

hdata2 = spcont.groupby(['continent', 'species','iucn', 'classify', 'systems'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).assign(ct =1).\
    groupby(['classify', 'systems'], as_index=False, dropna=False).\
    agg(spt = ('ct', sum)).\
    pivot_table(index= ['classify'], columns= 'systems', values='spt').\
    reset_index()

hdata2.to_csv('hdata2.csv')

#=====================================================================================
#2 Regional assessments for species
#==============================================================================
regdata = redata.replace(r'^\s*$', np.nan, regex=True).\
    dropna(subset=['regcont'], inplace=False).\
    groupby(['regcont','regass','species', 'iucncat', 'riucnwt'], as_index=False).\
    agg(tot = ('species', len)).\
    assign(cscomb = regdata.regcont+'_'+regdata.species).\
    filter(items =['cscomb','regass', 'regcont', 'iucncat', 'riucnwt'])

spmean = spdata2.groupby(['continent', 'species'], as_index=False, dropna=False).\
    agg(cts = ('species', len)).replace(r'^\s*$', np.nan, regex=True).\
    dropna(subset=['species'], inplace= False).merge(actdata, how='left', on=['species']).\
    assign(cscomb = spmean.continent+'_'+spmean.species).replace(np.nan, "NE", regex=True).\
    merge(regdata, how='left', on=['cscomb']).query('riucnwt.notnull()')

#=================
# compute the biodiversity congruence coefficient
bcc = spmean.\
    assign(giucnwt = np.select([spmean['iucnall'] == 'LC',
                                spmean['iucnall'] == 'NT',
                                spmean['iucnall'] == 'VU',
                                spmean['iucnall'] == 'EN',
                                spmean['iucnall'] == 'CR',
                                spmean['iucnall'] == 'DD',
                                spmean['iucnall'] == 'NE',
                                spmean['iucnall'] == 'EW',
                                spmean['iucnall'] == 'EX'],
                                [1, 2, 3, 4, 5, 5, 5, 6, 7], default = 0),
    spcc = (bcc.giucnwt)/(bcc.riucnwt)**2, ct=1)
bcc.to_csv('indicesdata1.csv', index=False)

sppr = bcc.groupby(['regass', 'species'], as_index=False).agg(tot = ('ct', sum)).\
    assign(cts = 1, sr = sppr.groupby(['regass'])['cts'].transform(sum)).\
    filter(items= ['regass', 'sr']).groupby('regass', as_index= False).\
    agg(r = ('sr', np.mean))

bccf = bcc.merge(sppr, how='left', on='regass').\
    assign(rtsr = bccf.spcc/bccf.r).\
    groupby(['regass'], as_index=False).agg(ccf = ('rtsr', sum)).\
    sort_values('ccf', ascending=False)

#============
#biodiversity confusion matrix

bcf = bcc.groupby(['regass', 'iucnall', 'iucncat'], as_index=False, dropna=False).\
    agg(tot = ('ct', len)).\
    assign(comcol = np.where(bcf['iucnall'] == bcf['iucncat'], "sclass", "dclass"),
           classd = bcf.groupby(['regass', 'comcol'])['tot'].transform(sum),
           classt = bcf.groupby(['regass'])['tot'].transform(sum))

bapr = bcf.query('comcol=="sclass"').assign(accuracy = (bcf.classd/bcf.classt)*100).\
    groupby('regass', as_index=False).\
    agg(accuf = ('accuracy', np.mean)).sort_values('accuf', ascending=False)



#relationship btn biodiversity accuracy confusion accuracy and congrunce coffeicient
bdata = bccf.merge(bapr, how='left', on = 'regass')
bdata.to_csv('combinedcoac1.csv')

g = sns.lmplot(y= 'accuf', x= 'ccf', data = bdata, scatter_kws={"color": "black"},
               line_kws={"color": "black"})

def annotate(data, **kws):
    r, p = st.pearsonr( data['ccf'], data['accuf'])
    ax = plt.gca()
    ax.text(.2, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
g.map_dataframe(annotate)
plt.legend()
plt.xlabel('Biodiversity congruence coefficient')
plt.ylabel('conservation similarity index')



#heat maps for the different regions
pvtdata = bcc.groupby(['regass', 'iucnRedListCategory', 'iucncat'], as_index=False, dropna=False).\
    agg(tot = ('ct', len)).\
    pivot_table(index= ['regass','iucnRedListCategory'], columns= 'iucncat', values='tot').\
    reset_index().fillna(0, inplace=False)

#==================
#species and country endemicity
#========
contr = spdata2.filter(items=['continent', 'countryCode']).assign(ct =1).\
    groupby(['continent', 'countryCode'], as_index=False).\
    agg(cout = ('ct', len)).assign(ct =1).\
    groupby(['continent'], as_index=False).\
    agg(cout = ('ct', len))


enddata = spdata2.assign(ct = 1).\
    groupby(['continent', 'countryCode', 'species'], as_index=False, dropna=False).\
    agg(cout = ('ct', len)).assign(ct = 1, tospecies = 'tspecies').\
    assign(ccount = enddata.groupby(['tospecies', 'species'])['ct'].transform(sum),
           ctryr = enddata.groupby(['countryCode', 'continent'])['ct'].transform(sum),
           contr = enddata.groupby(['continent', 'tospecies', 'species'])['ct'].transform(sum),
           cno = 247,
           endemcity = (1-enddata.ccount/enddata.cno)/enddata.ctryr).\
    query('endemcity.notnull()')

endemc = enddata.query('ccount==1').groupby('countryCode', as_index=False).agg(endt = ('ct', sum)).\
    sort_values('endt', ascending=False)
enddata.to_csv('enddata.csv', index=False)

endctry = enddata.groupby(['continent', 'countryCode'], as_index=False).agg(endt = ('endemcity', sum)).\
    sort_values('endt', ascending=False).query('countryCode !="ZZ"')

contd = enddata.groupby('continent', as_index=False).agg(endc = ('endemcity', sum)).\
    merge(contr, how='left', on='continent').\
    assign(endf = contd.endc/contd.cout,pct = contd.endf*100,
           conti= np.where(contd.continent == 'AFRICA', 'AF',
                           np.where(contd.continent == 'ASIA', 'AS',
                                    np.where(contd.continent == 'ANTARCTICA', 'AR',
                                             np.where(contd.continent == 'SOUTH_AMERICA', 'SA',
                                                      np.where(contd.continent == 'NORTH_AMERICA', 'NA',
np.where(contd.continent == 'OCEANIA', 'OC', np.where(contd.continent == 'EUROPE', 'EU', 'not'
                                             ))))))))

sns.boxplot(data=contend, x="continent", y="endemcity")
sns.despine(left=True)

mod = ols('endemcity ~ continent', data=enddata).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)

tukey = pairwise_tukeyhsd(endog=enddata['endemcity'],
                          groups=enddata['continent'],
                          alpha=0.05)
print(tukey)
#===========Graphical presentation
#world map from geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

cdatar= cdata.filter(items = ['name', 'richness'])

wdata = world.merge(cdatar, how='left', on='name')

fig, ax = plt.subplots(figsize = (8, 7))
grouped = wdata[['name', 'geometry', 'richness']]
countries = grouped.dissolve(by='name', aggfunc='mean')
world.plot(ax = ax)
countries.plot(ax= ax, column='richness', cmap='jet', scheme='quantiles', legend = True,
               legend_kwds={'loc': 'lower left', 'fmt':"{:.0f}", 'bbox_to_anchor':(0.02, 0.3)})
              # scheme='userdefined',
               #classification_kwds={'bins':[100, 10100, 20100, 30100, 40100, 50100]},
               #legend=True)
#=================
#level of endemicty
ccodes = cdata.filter(items=['countryCode', 'name'])
contend = enddata.groupby('continent', as_index=False).agg(endd = ('endemcity', sum))
cenddata = enddata.groupby(['continent','countryCode'], as_index=False).agg(endd = ('endemcity', sum)). \
    groupby(['countryCode'], as_index=False).agg(endd2=('endd', np.mean)).\
    sort_values(['endd2'], ascending=False).merge(ccodes, how='left', on='countryCode')
wdend = world.merge(cenddata, how='left', on='name').assign(endd3 = wdend.endd2)

fig, ax = plt.subplots(figsize = (8, 7))
wdend.plot(ax = ax, column='endd3', cmap='jet',legend=True,  scheme='quantiles', #scheme = 'userdefined',
           #classification_kwds={'bins':[78, 82, 86, 90, 94, 98, 102]},
           legend_kwds={'loc': 'lower left', 'fmt':"{:.3f}", 'bbox_to_anchor':(0.02, 0.2)})
fig.savefig("endemict.png", dpi = 600, bbox_inches='tight')
#lat, lon = 0.1432608, -0.133208
#ax_end = fig.add_axes([0.5*(1.2+lon/180) , 0.5*(0.5+lat/1) , 0.12, 0.12])
#ax_end.bar(x= contd.conti, height= contd.pct,
           #color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink'])
#plt.setp(ax_end.get_xticklabels(), rotation=45, horizontalalignment='right')
#ax_end.patch.set_facecolor('#E0E0E0')
#ax_end.patch.set_alpha(0.2)
#===========


#================
#collate the iucn redlait from IUCN database directly
iuncgbif = spdata.groupby(['species'], as_index=False).\
    agg(ct = ('species', len)).replace(r'^\s*$', np.nan, regex=True).\
    dropna(subset = ['species'], inplace=False).merge(actdata, how='left', on='species').\
    replace(np.nan, 'NE', regex=True).filter(items= ['species', 'iucnall'])



pdata = spdata2.groupby(['continent', 'species'], as_index=False).\
    agg(ct = ('species', len)).merge(iuncgbif, how='left', on='species').assign(ct = 1).\
    groupby(['continent', 'iucnall'], as_index=False).\
    agg(iucn = ('ct', sum)).sort_values('iucn', ascending=False)

contrich = pd.read_csv('conrichness.csv').filter(items=['continent', 'sprich'])


fig = plt.figure()
ax_map = fig.add_axes([0, 0, 1, 1])

grouped2 = wdata[['continent', 'geometry', 'richness']]
countinents = grouped2.dissolve(by='continent', aggfunc='sum').merge(contrich, how='left', on='continent')
countinents.plot(ax= ax_map, column='sprich', cmap='jet', scheme='quantiles', legend=True,
                 legend_kwds={'loc': 'lower left', 'fmt':"{:.0f}", 'bbox_to_anchor':(0.04, 0.5)})

lat, lon = 0.1432608, -0.133208
ax_pie = fig.add_axes([0.5*(0.08+lon/180) , 0.5*(0.2+lat/1) , 0.3, 0.3])
pied = pdata.groupby('iucnall', as_index=False).agg(toti = ('iucn', sum)).\
    assign(call = "total", pct = pied.toti/pied.groupby('call')['toti'].transform(sum)*100)
ax_pie.pie(pied.pct, autopct='%1.1f%%', pctdistance=1.2,
        labeldistance= 1,colors=['red', 'blue', 'green','orange', 'grey',
                                 'yellow','pink', 'olive', 'maroon'])
ax_pie.legend(labels= pied.iucnall, bbox_to_anchor = (0,1))
ax_pie.patch.set_facecolor('#E0E0E0')
plt.title('Global extinction threats', fontsize = 9)



lat, lon = 0.1432608, -0.133208
ax_bar7 = fig.add_axes([0.5*(1.56+lon/180) , 0.5*(0.5+lat/1) , 0.13, 0.15])
pafr = pdata.query('continent=="OCEANIA"')
ax_bar7.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar7.get_xticklabels():
    tick.set_rotation(45)
ax_bar7.patch.set_facecolor('#E0E0E0')
ax_bar7.patch.set_alpha(0.2)

lat, lon = 0.1432608, -0.133208
ax_bar6 = fig.add_axes([0.5*(1.1+lon/180) , 0.5*(0.1+lat/1) , 0.13, 0.15])
pafr = pdata.query('continent=="ANTARCTICA"')
ax_bar6.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar6.get_xticklabels():
    tick.set_rotation(45)
ax_bar6.tick_params(axis='x', colors='white')
ax_bar6.patch.set_facecolor('#E0E0E0')
ax_bar6.patch.set_alpha(0.2)


lat, lon = 0.1432608, -0.133208
ax_bar5 = fig.add_axes([0.5*(1.1+lon/180) , 0.5*(1.5+lat/1) , 0.13, 0.15])
pafr = pdata.query('continent=="EUROPE"')
ax_bar5.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar5.get_xticklabels():
    tick.set_rotation(45)
ax_bar5.patch.set_facecolor('#E0E0E0')
ax_bar5.tick_params(axis='x', colors='white')
ax_bar5.patch.set_alpha(0.2)

lat, lon = 0.1432608, -0.133208
ax_bar4 = fig.add_axes([0.5*(1.3+lon/180) , 0.5*(1.1+lat/1) , 0.13, 0.15])
pafr = pdata.query('continent=="ASIA"')
ax_bar4.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar4.get_xticklabels():
    tick.set_rotation(45)
ax_bar4.patch.set_facecolor('#E0E0E0')
ax_bar4.patch.set_alpha(0.2)


lat, lon = 31.432608, -120.133208
ax_bar = fig.add_axes([0.5*(1+lon/180) , 0.5*(1+lat/90) , 0.13, 0.15])
pafr = pdata.query('continent=="NORTH_AMERICA"')
ax_bar.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar.get_xticklabels():
    tick.set_rotation(45)
ax_bar.patch.set_facecolor('#E0E0E0')
ax_bar.patch.set_alpha(0.2)

lat, lon = 0.01432608, -77.133208
ax_bar2 = fig.add_axes([0.5*(1+lon/180) , 0.5*(0.8+lat/1) , 0.11, 0.13])
pafr = pdata.query('continent=="SOUTH_AMERICA"')
ax_bar2.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
for tick in ax_bar2.get_xticklabels():
    tick.set_rotation(45)
ax_bar2.patch.set_facecolor('#E0E0E0')
ax_bar2.patch.set_alpha(0.2)


lat, lon = 0.01432608, -10.133208
ax_bar3 = fig.add_axes([0.5*(1+lon/180) , 0.5*(0.9+lat/1) , 0.13, 0.15])
pafr = pdata.query('continent=="AFRICA"')
ax_bar3.bar(x= pafr.iucnall, height= pafr.iucn,
           color=['red', 'blue', 'green','orange', 'grey', 'yellow','pink', 'gold', 'maroon'])
#plt.grid(False)
#plt.axis('off')
for tick in ax_bar3.get_xticklabels():
    tick.set_rotation(45)
ax_bar3.patch.set_facecolor('#E0E0E0')
ax_bar3.patch.set_alpha(0.2)

###threats vs endemicity vs richness
thindex = spdata2.groupby(['countryCode', 'species'], as_index=False, dropna=False).\
    agg(cts = ('species', len)).replace(r'^\s*$', np.nan, regex=True).\
    dropna(subset=['species'], inplace= False).merge(actdata, how='left', on=['species']).\
    replace(np.nan, "NE" , regex=True).\
    assign(ct = 1).groupby(['countryCode', 'iucnall'], as_index=False, dropna = False).\
    agg(tot = ('ct', sum)).\
    assign(giucnwt = np.select([thindex['iucnall'] == 'LC',
                                thindex['iucnall'] == 'NT',
                                thindex['iucnall'] == 'VU',
                                thindex['iucnall'] == 'EN',
                                thindex['iucnall'] == 'CR',
                                thindex['iucnall'] == 'DD',
                                thindex['iucnall'] == 'NE',
                                thindex['iucnall'] == 'EW',
                                thindex['iucnall'] == 'EX'],
                                [1, 2, 3, 4, 5, 5, 5, 6, 7], default = 0),
           iucnt = thindex.giucnwt*thindex.tot,
           richness = thindex.groupby('countryCode')['tot'].transform(sum)).\
    groupby('countryCode', as_index=False, dropna=False).\
    agg(richn = ('richness', np.mean), totwt = ('iucnt', sum))

suicn = thindex.assign(cs = thindex.totwt/thindex.richn).filter(items=['countryCode', 'cs'])

endrich = enddata.groupby('countryCode', as_index=False).\
    agg(S = ('ctryr', np.mean), ec = ('endemcity', sum)).\
    merge(suicn, how='left', on='countryCode').query('countryCode != ["ZZ","RU"]')

mask  = np.triu(np.ones_like(endrich.corr()))
sns.heatmap(endrich.corr(), annot =True, mask = mask, cbar = False)

#im = plt.imread('E:\\PhD\\manuscript\\extinction\\data analysis/graphs/rela.png')

fig2 = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.scatter3D(endrich.S, endrich.ec, endrich.cs,
              c=endrich.ec+endrich.cs+endrich.S, cmap='viridis')
newax = fig2.add_axes([0.45,0.25,0.2,0.2], anchor='NE', zorder=1)
sns.heatmap(endrich.corr(), annot =True, mask = mask, cbar = True, ax = newax)
newax.patch.set_facecolor('#E0E0E0')
newax.patch.set_alpha(0.4)
plt.title('Correlations', fontsize = 9)
ax3.set_xlabel('Richness (S)', rotation=45 , ha='right')
ax3.set_ylabel('Endemicity coefficient (ec)', rotation=45)
ax3.set_zlabel('Conservation score (cs)', rotation=45)
#ax3.tick_params(axis='z', which='major', pad=0.09)

fig2.savefig("correlation.png", dpi = 600, bbox_inches='tight')
