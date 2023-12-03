# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:24:37 2023

@author: TAMILSELVAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



def read_process_clean(filename, indicator_name):
    world = pd.read_csv(filename, skiprows=3)
    country = ['China', 'United Kingdom', 'India',
               'Indonesia', 'Pakistan', 'United States']
    data = world[(world['Indicator Name'] == indicator_name)
                 & (world['Country Name'].isin(country))]
    sel_data = data.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
                          '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
                          '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
                          '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
                          '1996', '1997', '1998', '1999', '2000', '2021', '2022', 'Unnamed: 67'], axis=1).reset_index(drop=True)
    sel_t = sel_data.transpose()
    sel_t.columns = sel_t.iloc[0]
    sel_t = sel_t.iloc[1:]
    sel_t.index = pd.to_numeric(sel_t.index)
    sel_t['Years'] = sel_t.index
    return sel_data, sel_t


def slice_data(df):
    df = df[['Country Name', '2020']]
    return df


def merge_five(x1, x2, x3, x4, x5):
    merge1 = pd.merge(x1, x2, on='Country Name', how='outer')
    merge2 = pd.merge(merge1, x3, on='Country Name', how='outer')
    merge3 = pd.merge(merge2, x4, on='Country Name', how='outer')
    merge4 = pd.merge(merge3, x5, on='Country Name', how='outer')
    merge4 = merge4.reset_index(drop=True)
    return merge4


def plot_heatmap(df):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), cmap='RdYlGn', square=True,
                linewidths=.5, annot=True, fmt=".2f", center=0)
    plt.title("Correlation matrix of Indicators")
    plt.savefig('heatmap.png')
    plt.show()

def create_dotplot(df, title):
    sns.set_style("whitegrid")
    dot = sns.catplot(x = 'Years', y = 'Value', hue = 'Country', data = df.melt(id_vars=['Years'], var_name='Country', value_name='Value'), kind="point", ylabel='kilo')
    dot.set_xticklabels(rotation=90)
    plt.title(title)
    plt.ylabel('kilotonns')
    plt.savefig('dotplot.png')
    plt.show()


def create_barplot(df, x_value, y_value, head_title, x_label, y_label, colors, figsize=(10, 6)):
    sns.set_style('whitegrid')
    df.plot(x=x_value, y=y_value, kind='bar', title=head_title, color=colors,
            width=0.65, figsize=figsize, xlabel=x_label, ylabel=y_label)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.savefig('barplot.png')
    plt.show()


def create_boxplot(df, countries, shownotches=True):
    plt.figure(figsize=(10, 5))
    plt.boxplot([df[country] for country in countries])
    plt.title('Cereal production of selected countries')
    plt.xticks(range(1, len(countries) + 1), countries)
    plt.savefig('boxplot.png')
    plt.show()


def create_lineplot(df, y_label, title):
    sns.set_style("whitegrid")
    df.plot(x='Years', y=['China', 'United Kingdom', 'Indonesia', 'India',
            'Pakistan', 'United States'], xlabel='Years', ylabel=y_label, marker='.')
    plt.title(title)
    plt.xticks(range(2000, 2021, 2))
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.savefig('lineplot.png')
    plt.show()


def create_pieplot(df, Years, autopct='%1.0f%%', fontsize=11):
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    label = ['China', 'United Kingdom', 'Indonesia',
             'India', 'Pakistan', 'United States']
    plt.figure(figsize=(4, 5))
    plt.pie(df[str(Years)],
            autopct=autopct, labels=label, explode=explode,
            startangle=180, wedgeprops={"edgecolor": "black", "linewidth": 2, "antialiased": True},)
    plt.title(f'Population in {Years}', fontsize=fontsize)
    plt.savefig('pieplot.png')
    plt.show()
    
    
def skew_kurt_plot(data):
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"skewness :{skewness}")
    print(f"kurtosis :{kurtosis}")
    plt.figure(figsize=(7,4))
    plt.hist(data, bins='auto', alpha=0.7, color= 'green', edgecolor='black')
    plt.grid(True)
    plt.title(f"Histogram of China's Total CO2 Emission from 2001 to 2020, skewness={skewness:.2f} and kurtosis={kurtosis:.2f}")
    plt.xlabel("China's Total CO2 Emission from 2001 to 2020")
    plt.ylabel('Frequency')
    plt.savefig('histplot.png')
    plt.show()


co2 , co2_t  = read_process_clean('API_19_DS2_en_csv_v2_5998250.csv', 'CO2 emissions (kt)')
frwt, frwt_t = read_process_clean('API_19_DS2_en_csv_v2_5998250.csv', 'Annual freshwater withdrawals, total (billion cubic meters)')
cere, cere_t = read_process_clean('API_19_DS2_en_csv_v2_5998250.csv', 'Cereal yield (kg per hectare)')
acce, acce_t = read_process_clean('API_19_DS2_en_csv_v2_5998250.csv', 'Access to electricity (% of population)')
popu, popu_t = read_process_clean('API_19_DS2_en_csv_v2_5998250.csv', 'Population, total')

co2_cor = slice_data(co2).rename(columns={'2020': 'CO2_emission'})
frwt_cor = slice_data(frwt).rename(columns={'2020': 'Freshwater_withdrawals'})
cere_cor = slice_data(cere).rename(columns={'2020': 'Cereal_production'})
acce_cor = slice_data(acce).rename(columns={'2020': 'Access to electricity'})
popu_cor = slice_data(popu).rename(columns={'2020': 'Total Population'})

co2_frwt_cere_acce_popu = merge_five( co2_cor,frwt_cor, cere_cor, acce_cor, popu_cor)


print(co2_frwt_cere_acce_popu.describe())

plot_heatmap(co2_frwt_cere_acce_popu)

create_dotplot(co2_t, 'CO2 Emission')

bar_frwt = frwt_t[frwt_t['Years'].isin([2001, 2005, 2010, 2015, 2020])]

create_barplot(bar_frwt, 'Years', ['China', 'United Kingdom', 'Indonesia','India', 'Pakistan', 'United States'], 'Freshwater Withdrawals in Barplot','Years', 'kilotons', ('lightgreen', 'lightblue','violet','black','red','yellow'))

create_boxplot(cere_t, ['China', 'United Kingdom', 'Indonesia', 'India', 'Pakistan', 'United States'], shownotches=True)

create_lineplot(acce_t,'% of population', 'Access to electricity')

create_pieplot(popu, 2020)

skew_kurt_plot(co2_t['China'])
