# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:24:37 2023

@author: TAMILSELVAN
"""
# import and function

import pandas as pd  # Inputing file (eg, pd.read_csv), Data-processing
import matplotlib.pyplot as plt  # Visualisation
import seaborn as sns  # Visualisation
import scipy.stats as stats  # Statistical variable


def read_process_clean(filename, indicator_name):
    '''Reading the raw data, Processing to check any Nan value presented in 
my data and selecting some countries for our experiment, Cleaning to 
drop Nan value and Transposing the dataframe.Years from 2001 to 2020.
Countries are also selected in upcoming code.'''

    world = pd.read_csv(filename, skiprows=3)
    country = ['China', 'United Kingdom', 'India',
               'Indonesia', 'Pakistan', 'United States']
    data = world[(world['Indicator Name'] == indicator_name)
                 & (world['Country Name'].isin(country))]
    columns_to_drop = ['Country Code', 'Indicator Name',
                       'Indicator Code', 'Unnamed: 67']
    sel_data = data.drop(columns_to_drop + [
        str(year) for year in range(1960, 2001)] + [
        str(year) for year in range(2021, 2023)], axis=1)
    cleaned_data = sel_data.reset_index(drop=True)

    cleaned_data_t = cleaned_data.transpose()
    cleaned_data_t.columns = cleaned_data_t.iloc[0]
    cleaned_data_t = cleaned_data_t.iloc[1:]
    cleaned_data_t.index = pd.to_numeric(cleaned_data_t.index)
    cleaned_data_t['Years'] = cleaned_data_t.index
    cleaned_data_t.reset_index(drop=True)
    return cleaned_data, cleaned_data_t


def slice_data(df):
    '''Slicing the data for correlation'''

    df = df[['Country Name', '2020']]
    return df


def merge_five(x1, x2, x3, x4, x5):
    '''Merging 5 dataframe of sliced data for Correlation'''

    merge1 = pd.merge(x1, x2, on='Country Name', how='outer')
    merge2 = pd.merge(merge1, x3, on='Country Name', how='outer')
    merge3 = pd.merge(merge2, x4, on='Country Name', how='outer')
    merge4 = pd.merge(merge3, x5, on='Country Name', how='outer')
    merge4 = merge4.reset_index(drop=True)
    return merge4


def create_heatmap(df):
    '''Defining the function to create the heatmap by indicator name wise,
and giving the suitable title. The labels are not given because 
the X and Y values are same Indicator name'''

    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), cmap='RdYlGn', square=True,
                linewidths=.5, annot=True, fmt=".2f", center=0)
    plt.title("Correlation matrix of selected Indicators by the year 2020")
    plt.savefig('heatmap.png')
    plt.show()


def create_dotplot(df, title):
    '''Defining the function to create the dotplot,then applyig the dataframe
into the function, labelling the X-axis and Y-axis, Naming the dotplot,
Create legend in best place'''

    sns.set_style("whitegrid")
    dot = sns.catplot(x='Years', y='Value', hue='Country',
                      data=df.melt(id_vars=['Years'], var_name='Country',
                                   value_name='Value'), kind="point",
                      ylabel='kilotons')
    dot.set_xticklabels(rotation=90)
    plt.title(title)
    plt.savefig('dotplot.png')
    plt.show()


def create_barplot(df, x_value, y_value, head_title, x_label, y_label, colors):
    '''Defining the function to create the barplot to comparing the selected
countries, then applyig the dataframe into the function, labelling the X-axis
and Y-axis, Naming the barplot, Create legend in best place'''

    sns.set_style('whitegrid')
    df.plot(x=x_value, y=y_value, kind='bar', title=head_title, color=colors,
            width=0.65, figsize=(10, 6), xlabel=x_label, ylabel=y_label)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.savefig('barplot.png')
    plt.show()


def create_boxplot(data, countries):
    '''Defining the function to create the boxplot,then applyig the dataframe
into the function, Naming the boxplot.'''

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame({country : data[country] for country in countries})
    sns.boxplot(data=df)
    plt.title('Cereal production of selected countries')
    plt.xlabel('Country Name')
    plt.ylabel('Values') 
    plt.show()


def create_lineplot(df, y_label, title):
    '''Defining the function to create the lineplot,then applyig the dataframe
into the function, labelling the X-axis and Y-axis, Naming the lineplot,
Create legend in best place'''

    sns.set_style("whitegrid")
    df.plot(x='Years', y=['China', 'United Kingdom', 'Indonesia', 'India',
            'Pakistan', 'United States'], xlabel='Years', ylabel=y_label,
            marker='.')
    plt.title(title)
    plt.xticks(range(2000, 2021, 2))
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    plt.savefig('lineplot.png')
    plt.show()


def create_pieplot(df, Years, autopct='%1.0f%%', fontsize=11):
    '''Using pieplot to compare the percentage of all seleted Countries 
Population total year 2001 and 2020, which is the dataframe of 
before transposing '''

    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    label = ['China', 'United Kingdom', 'Indonesia',
             'India', 'Pakistan', 'United States']
    plt.figure(figsize=(4, 5))
    plt.pie(df[str(Years)],
            autopct=autopct, labels=label, explode=explode,
            startangle=180, wedgeprops={"edgecolor": "black", "linewidth": 2,
                                        "antialiased": True},)
    plt.title(f'Population % in {Years}', fontsize=fontsize)
    plt.savefig('pieplot.png')
    plt.show()


def skew_kurt_plot(data):
    '''Creating a function for histogram and to check the skewness and 
kurtosis value of required data, X and Y labels are created, the suitable 
title for the histogram is also mentioned'''

    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"skewness :{skewness}")
    print(f"kurtosis :{kurtosis}")
    plt.figure(figsize=(7, 4))
    plt.hist(data, bins='auto', alpha=0.7, color='green', edgecolor='black')
    plt.grid(True)
    title_components = [
        "Histogram of China's Total CO2 Emission from 2001 to 2020",
        f"Skewness={skewness:.2f} and Kurtosis={kurtosis:.2f}"
    ]
    title = '\n'.join(title_components)
    plt.title(title)
    plt.xlabel("China's Total CO2 Emission from 2001 to 2020")
    plt.ylabel('Frequency')
    plt.savefig('histplot.png')
    plt.show()


# Main programme
'''Calling two dataframe (normal and transposed) by mentioning their
respective Indicators Name'''

filename = 'API_19_DS2_en_csv_v2_5998250.csv'

co2, co2_t = read_process_clean(filename, 'CO2 emissions (kt)')

frwt, frwt_t = read_process_clean(
    filename, 'Annual freshwater withdrawals, total (billion cubic meters)')

cere, cere_t = read_process_clean(filename, 'Cereal yield (kg per hectare)')

acce, acce_t = read_process_clean(filename,
                                  'Access to electricity (% of population)')

popu, popu_t = read_process_clean(filename, 'Population, total')

# The dataframe of sliced data for correlation
co2_cor = slice_data(co2).rename(columns={'2020': 'CO2_emission'})
frwt_cor = slice_data(frwt).rename(columns={'2020': 'Freshwater_withdrawals'})
cere_cor = slice_data(cere).rename(columns={'2020': 'Cereal_production'})
acce_cor = slice_data(acce).rename(columns={'2020': 'Access to electricity'})
popu_cor = slice_data(popu).rename(columns={'2020': 'Total Population'})

# Merging the above 5 sliced data
co2_frwt_cere_acce_popu = merge_five(
    co2_cor, frwt_cor, cere_cor, acce_cor, popu_cor)

'''Calling the function describe(), to know the value of Mean, Median, Count,
 Standard deviation, Quartile, Minimum, Maximum'''
print(co2_frwt_cere_acce_popu.describe())


# Visualisation 1 (Heatmap)
create_heatmap(co2_frwt_cere_acce_popu)

# Visualisation 2 (Dotplot)
create_dotplot(co2_t, 'CO2 Emission in kilotons ')

'''Only suggesting 5 years to show in Barplot of all selected countries'''
bar_frwt = frwt_t[frwt_t['Years'].isin([2001, 2005, 2010, 2015, 2020])]

# Visualisation 3 (Barplot)
create_barplot(bar_frwt, 'Years', ['China', 'United Kingdom', 'Indonesia',
                                   'India', 'Pakistan', 'United States'],
               'Freshwater Withdrawals in Barplot from year 2001 to 2020',
               'Years', 'kilotons', ('lightgreen', 'lightblue', 'violet',
                                     'black', 'red', 'yellow'))

# Visualisation 4 (Boxplot)
create_boxplot(cere_t, ['China', 'United Kingdom', 'Indonesia',
               'India', 'Pakistan', 'United States'])

# Visualisation 5 (Lineplot)
create_lineplot(acce_t, '% of population',
                'Access to electricity % of population from 2001 to 2020 year')

# Visualisation 6 (Pieplot)
create_pieplot(popu, 2020)

'''Analysing skewness and kurtosis for large CO2 Emission producing
 Country China and creating histogram of the respective data'''
# Visualisation 7 (Histogram)
skew_kurt_plot(co2_t['China'])
