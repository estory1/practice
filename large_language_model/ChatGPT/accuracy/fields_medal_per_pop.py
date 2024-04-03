#%%[markdown]
## Fields Medals per-capita, per nation
#
# Date created: 20240402
#
# **Motivating claim**: that [the US is a nation skilled at math, as evidenced by having claimed the most Fields Medals of any nation](https://twitter.com/ToKTeacher/status/1774363133152469291).
#
# Can Americans "do math"? Sure, the US has collected the largest count of Fields Medals - but the US is a relatively populous country. All else equal, we should expect that greater population will yield greater numbers of Fields Medal winners.
#
# Is all else *really* equal, though?
#
# A first-step towards a fairer international comparison would occur per-capita. Let's dig-in!
#
# **Pre-reqs:**
# - Python and pandas.
# - To extract Markdown tables from HTML tables (but unfortunately not DIV-defined tables), you need:
#   1. A Chromium-compatible web browser (Google Chrome, Brave, Microsoft Edge, Chromium, etc.).
#   2. Install this extension: [HTML Table to Markdown](https://chromewebstore.google.com/detail/html-table-to-markdown/ghcdpakfleapaahmemphphdojhdabojj)
#   3. With that^ extension installed, open DevTools:
#       - On Mac: Cmd+Option+I
#       - On Windows: Ctrl+Shift+I
#   4. Go to the "Elements" tab and left-click on the table you want to convert.
#   5. Then click the Table to Markdown tab.
#   6. Click "Convert".

#%%[markdown]
### Fields Medal data extracted from the table at: [Fields Medal for Mathematics By Nation](https://stats.areppim.com/stats/stats_fieldsxnation.htm)
#
# |                                                      Country ¹                                                       |                                                        Awards                                                        |                                                   Percent of Total                                                   |
# |----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
# |                                                      Australia                                                       |                                                          2                                                           |                                                        2.90%                                                         |
# |                                                       Austria                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                       Belgium                                                        |                                                          2                                                           |                                                        2.90%                                                         |
# |                                                        Brazil                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        Canada                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        China                                                         |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                       Finland                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        France                                                        |                                                          13                                                          |                                                        18.84%                                                        |
# |                                                       Germany                                                        |                                                          2                                                           |                                                        2.90%                                                         |
# |                                                         Iran                                                         |                                                          2                                                           |                                                        2.90%                                                         |
# |                                                        Israel                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        Italy                                                         |                                                          2                                                           |                                                        2.90%                                                         |
# |                                                        Japan                                                         |                                                          3                                                           |                                                        4.35%                                                         |
# |                                                     New Zealand                                                      |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        Norway                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        Russia                                                        |                                                          6                                                           |                                                        8.70%                                                         |
# |                                                     Soviet Union                                                     |                                                          3                                                           |                                                        4.35%                                                         |
# |                                                        Sweden                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                       Ukraine                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                    United Kingdom                                                    |                                                          8                                                           |                                                        11.59%                                                        |
# |                                                    United States                                                     |                                                          15                                                          |                                                        21.74%                                                        |
# |                                                       Vietnam                                                        |                                                          1                                                           |                                                        1.45%                                                         |
# |                                                        Total                                                         |                                                          69                                                          |                                                       100.00%                                                        |
# | ¹ Double nationals are accounted for twice, once for each nationality. For further details, see List - Fields Medal. | ¹ Double nationals are accounted for twice, once for each nationality. For further details, see List - Fields Medal. | ¹ Double nationals are accounted for twice, once for each nationality. For further details, see List - Fields Medal. |

#%%[markdown]
### Let's start by asking ChatGPT to do this analysis for us.
#
#### Here's my ChatGPT log, having it perform some arithmetic on the data above:

#%%[markdown]
# **User**
# The following is a table of country, awards, and percent of total. Ignore the percent column. Divide the awards for a country by that country's population, then return the resulting list of awards per country population:
#
# ```
# Fields Medal for Mathematics
# By Nation, 1936 - 2022
# Country ¹	Awards	Percent of Total
# Australia	2	2.90%
# Austria	1	1.45%
# Belgium	2	2.90%
# Brazil	1	1.45%
# Canada	1	1.45%
# China	1	1.45%
# Finland	1	1.45%
# France	13	18.84%
# Germany	2	2.90%
# Iran	2	2.90%
# Israel	1	1.45%
# Italy	2	2.90%
# Japan	3	4.35%
# New Zealand	1	1.45%
# Norway	1	1.45%
# Russia	6	8.70%
# Soviet Union	3	4.35%
# Sweden	1	1.45%
# Ukraine	1	1.45%
# United Kingdom	8	11.59%
# United States	15	21.74%
# Vietnam	1	1.45%
# Total	69	100.00%
# ```
# **ChatGPT**
# To calculate the awards per capita for each country, we need the population data for each country. Since the table doesn't include population information, I'll make use of approximate population data based on the latest available statistics. Please note that these population figures are estimates and may not be entirely accurate.
#
# Here's the approximate population data (in millions) for each country:
# ```
# Australia: 25.7
# Austria: 8.9
# Belgium: 11.6
# Brazil: 213.9
# Canada: 38.1
# China: 1441.7
# Finland: 5.5
# France: 65.4
# Germany: 83.2
# Iran: 84.8
# Israel: 8.9
# Italy: 60.4
# Japan: 126.3
# New Zealand: 5.1
# Norway: 5.4
# Russia: 145.9
# Soviet Union: (not applicable)
# Sweden: 10.4
# Ukraine: 43.7
# United Kingdom: 68.2
# United States: 331.4
# Vietnam: 97.3
# ```
# Now, let's calculate the awards per capita for each country by dividing the number of awards by the population:
# ```
# Australia: 2 / 25.7 = 0.0779
# Austria: 1 / 8.9 = 0.1124
# Belgium: 2 / 11.6 = 0.1724
# Brazil: 1 / 213.9 = 0.0047
# Canada: 1 / 38.1 = 0.0262
# China: 1 / 1441.7 = 0.0007
# Finland: 1 / 5.5 = 0.1818
# France: 13 / 65.4 = 0.1986
# Germany: 2 / 83.2 = 0.0240
# Iran: 2 / 84.8 = 0.0236
# Israel: 1 / 8.9 = 0.1124
# Italy: 2 / 60.4 = 0.0331
# Japan: 3 / 126.3 = 0.0237
# New Zealand: 1 / 5.1 = 0.1961
# Norway: 1 / 5.4 = 0.1852
# Russia: 6 / 145.9 = 0.0411
# Soviet Union: (not applicable)
# Sweden: 1 / 10.4 = 0.0962
# Ukraine: 1 / 43.7 = 0.0229
# United Kingdom: 8 / 68.2 = 0.1172
# United States: 15 / 331.4 = 0.0453
# Vietnam: 1 / 97.3 = 0.0103
# ```
# Here are the awards per capita for each country rounded to four decimal places.

#%%[markdown]
### Oh - ChatGPT didn't return data for the Soviet Union. Let's ask ChatGPT to fix that.

#%%[markdown]
# **User**
#
# Repeat, but using the most-recent population value to impute for the Soviet Union.
#
# **ChatGPT**
#
# Since the Soviet Union dissolved in 1991, I'll use the population of the successor state, Russia, to estimate the population of the former Soviet Union.
#
# The population of Russia is approximately 145.9 million. Therefore, I'll use this value as an estimate for the population of the Soviet Union.
#
# Now, let's recalculate the awards per capita for each country:
# ```
# Australia: 2 / 25.7 = 0.0779
# Austria: 1 / 8.9 = 0.1124
# Belgium: 2 / 11.6 = 0.1724
# Brazil: 1 / 213.9 = 0.0047
# Canada: 1 / 38.1 = 0.0262
# China: 1 / 1441.7 = 0.0007
# Finland: 1 / 5.5 = 0.1818
# France: 13 / 65.4 = 0.1986
# Germany: 2 / 83.2 = 0.0240
# Iran: 2 / 84.8 = 0.0236
# Israel: 1 / 8.9 = 0.1124
# Italy: 2 / 60.4 = 0.0331
# Japan: 3 / 126.3 = 0.0237
# New Zealand: 1 / 5.1 = 0.1961
# Norway: 1 / 5.4 = 0.1852
# Russia: 6 / 145.9 = 0.0411
# Soviet Union: 3 / 145.9 = 0.0206 (using the population of Russia)
# Sweden: 1 / 10.4 = 0.0962
# Ukraine: 1 / 43.7 = 0.0229
# United Kingdom: 8 / 68.2 = 0.1172
# United States: 15 / 331.4 = 0.0453
# Vietnam: 1 / 97.3 = 0.0103
# ```
# Here are the awards per capita for each country, rounded to four decimal places, with the population of the Soviet Union imputed as the population of Russia.



#%% [markdown]
#### Looks good at first-glance! But [ChatGPT is known to do math incorrectly](https://duckduckgo.com/?q=chatgpt+incorrect+math) (particularly early in its public release, [in early 2023 - even of basic arithmetic](https://community.openai.com/t/chatgpt-calculations-are-wrong/37343), prior to [ChatGPT's plugin integration with Wolfram Alpha](https://writings.stephenwolfram.com/2023/03/chatgpt-gets-its-wolfram-superpowers/)).
## So we are wise not to accept ChatGPT's outputs on blind faith -- we should, at this moment in time, ideally "verify, don't trust" (instead of "trust, but verify" - or most naïvely of all: "trust, don't verify").
#### How accurate are its calculations? Let's find out.

#%%
import pandas as pd
from io import StringIO
import re

pd.options.display.expand_frame_repr = True
pd.options.display.max_rows = 100
pd.options.display.max_columns = 50


#%%
# copy-pasted text from ChatGPT
data_cgpt_raw = """Australia: 2 / 25.7 = 0.0779
Austria: 1 / 8.9 = 0.1124
Belgium: 2 / 11.6 = 0.1724
Brazil: 1 / 213.9 = 0.0047
Canada: 1 / 38.1 = 0.0262
China: 1 / 1441.7 = 0.0007
Finland: 1 / 5.5 = 0.1818
France: 13 / 65.4 = 0.1986
Germany: 2 / 83.2 = 0.0240
Iran: 2 / 84.8 = 0.0236
Israel: 1 / 8.9 = 0.1124
Italy: 2 / 60.4 = 0.0331
Japan: 3 / 126.3 = 0.0237
New Zealand: 1 / 5.1 = 0.1961
Norway: 1 / 5.4 = 0.1852
Russia: 6 / 145.9 = 0.0411
Soviet Union: 3 / 145.9 = 0.0206 (using the population of Russia)
Sweden: 1 / 10.4 = 0.0962
Ukraine: 1 / 43.7 = 0.0229
United Kingdom: 8 / 68.2 = 0.1172
United States: 15 / 331.4 = 0.0453
Vietnam: 1 / 97.3 = 0.0103"""

# data cleaning function
def clean_chatgpt(resp: str) -> str:
    return resp.replace(" (using the population of Russia)", "") \
        .replace(':', ',') \
        .replace('=', ',') \
        .replace('/',',') \
        .replace(" ", "")

# clean ChatGPT's response
data_cgpt = clean_chatgpt(data_cgpt_raw)
print(data_cgpt)   # cleaned CSV string, to spot-check QC

# read the CSV string into a DF
df = pd.read_csv(StringIO(data_cgpt), sep=',', names=["nat","fld","n","chatgpt_fld_per_megacapita"])
df

#%%
# do our own arithmetic, to the same 4 decimal places
df["our_fld_per_megacapita"] = df["fld"] / df["n"]
df["our_fld_per_megacapita_rounded"] = round(df["our_fld_per_megacapita"], 4)
df

#%%
# what's the (actual - expected) delta, as a % of ChatGPT's calculation?
df["chatgpt_error_pct"] = (df["chatgpt_fld_per_megacapita"] - df["our_fld_per_megacapita"]) / df["chatgpt_fld_per_megacapita"] * 100
df



#%%[markdown]
### Continuing with ChatGPT...

#%%[markdown]
# **User**
#
# Good enough! Now, instead of dividing by current populations, let's divide by the average population size of each nation since the Fields Medal was first awarded. Also, to simplify the analysis, let's combine Russia and Soviet Union, to treat those as 1 country for the duration of the medal's life.
#
# **ChatGPT**
#
# To calculate the average population size for each country since the Fields Medal was first awarded in 1936, we need historical population data. Since it's impractical to provide a complete historical dataset here, I'll use an estimation method to approximate the average population size for each country over this period.
#
# Additionally, I'll combine the population of Russia and the Soviet Union into one entity for the entire duration of the Fields Medal's existence.
#
# Here's a rough estimation of the average population size for each country over the period since 1936:
#```
# Australia: 12.6 million (average of historical data)
# Austria: 7.5 million (average of historical data)
# Belgium: 9.9 million (average of historical data)
# Brazil: 107.2 million (average of historical data)
# Canada: 22.6 million (average of historical data)
# China: 769.3 million (average of historical data)
# Finland: 4.7 million (average of historical data)
# France: 52.1 million (average of historical data)
# Germany: 69.2 million (average of historical data)
# Iran: 27.5 million (average of historical data)
# Israel: 5.5 million (average of historical data)
# Italy: 45.8 million (average of historical data)
# Japan: 102.1 million (average of historical data)
# New Zealand: 3.2 million (average of historical data)
# Norway: 4.0 million (average of historical data)
# Russia/Soviet Union: 134.8 million (average of historical data)
# Sweden: 7.8 million (average of historical data)
# Ukraine: 43.9 million (average of historical data)
# United Kingdom: 56.5 million (average of historical data)
# United States: 194.3 million (average of historical data)
# Vietnam: 44.1 million (average of historical data)
#```
# Now, let's recalculate the awards per capita for each country using these average population sizes:
#```
# Australia: 2 / 12.6 = 0.1587
# Austria: 1 / 7.5 = 0.1333
# Belgium: 2 / 9.9 = 0.2020
# Brazil: 1 / 107.2 = 0.0093
# Canada: 1 / 22.6 = 0.0442
# China: 1 / 769.3 = 0.0000
# Finland: 1 / 4.7 = 0.2128
# France: 13 / 52.1 = 0.2494
# Germany: 2 / 69.2 = 0.0289
# Iran: 2 / 27.5 = 0.0727
# Israel: 1 / 5.5 = 0.1818
# Italy: 2 / 45.8 = 0.0437
# Japan: 3 / 102.1 = 0.0294
# New Zealand: 1 / 3.2 = 0.3125
# Norway: 1 / 4.0 = 0.2500
# Russia/Soviet Union: 9 / 134.8 = 0.0668
# Sweden: 1 / 7.8 = 0.1282
# Ukraine: 1 / 43.9 = 0.0228
# United Kingdom: 8 / 56.5 = 0.1416
# United States: 15 / 194.3 = 0.0773
# Vietnam: 1 / 44.1 = 0.0227
#```
# Here are the awards per capita for each country, rounded to four decimal places, using the average population size since the Fields Medal was first awarded and combining Russia and the Soviet Union.

#%%[markdown]
### Wow, it really gave us the average populations for an arbitrary date range? And then recalculated?? Extremely impressive.
#### So impressive that we should question its correctness!
#
#### Let's try to verify via spot-check for the US.
#
# Let's verify-don't-trust against the US Census: [Historical Population Change Data (1910-2020)](https://www.census.gov/data/tables/time-series/dec/popchange-data-text.html)

#%%
# 
population_us_markdown = """|   State or Region    |     2020 Census      |     2010 Census      |     2000 Census      |     1990 Census      |     1980 Census      |     1970 Census      |     1960 Census      |     1950 Census      |     1940 Census      |     1930 Census      |     1920 Census      |     1910 Census      |
|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
|    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |    United States     |
| Resident Population  |     331,449,281      |     308,745,538      |     281,421,906      |     248,709,873      |     226,545,805      |     203,211,926      |     179,323,175      |     151,325,798      |     132,165,129      |     123,202,660      |     106,021,568      |      92,228,531      |"""

df_population_us = pd.read_table(StringIO(
            population_us_markdown.replace(",","")
        ), sep="|")
df_population_us = df_population_us.iloc[2].T.dropna()  # select just the Resident Population row, transpose, then drop rows having null
df_population_us = df_population_us[1:].astype(int)     # drop the first row (which is just the Resident Population label), then convert to int
df_population_us = pd.DataFrame({
    "census_year": df_population_us.index.str.replace(" Census","").astype(int), 
    "resident_population": df_population_us}, 
    index=df_population_us.index)
print(df_population_us.dtypes)
df_population_us
#%%
# calc the average US population over the years starting in 1940, since the first Field Medal was awarded in 1936: [Fields Medal - Wikipedia](https://en.wikipedia.org/wiki/Fields_Medal)
population_mean_us = df_population_us[ df_population_us["census_year"] >= 1936 ]["resident_population"].mean().astype(int)
print(f'mean resident_population (Census year >= 1936): {population_mean_us}')


#%%[markdown]
### Comparison of US population values returned by 3 methods:
# - **ChatGPT: 194.3M** (see above)
#
# - **Wolfram Alpha: 208.2M**: [average population size usa since 1936 - Wolfram|Alpha](https://www.wolframalpha.com/input?i=average+population+size+usa+since+1936)
# 
# - **Our Census calculation: 229.2M**
#
#### So, we find: at-most a `abs(229.2/194.3 - 1) * 100 = 17.96%` difference between ChatGPT's calculation and ours.
### Is this "too large"? Probably not for this exercise.
#
### But let's check another nation, France.

#%%
population_france_markdown = """| Year  | Population | Year | Population | Year | hide Population |
|-------|------------|------|------------|------|-----------------|
| 50 BC | 2,500,000  | 1806 | 29,648,000 | 1896 |   40,158,000    |
|   1   | 5,500,000  | 1811 | 30,271,000 | 1901 |   40,681,000    |
|  120  | 7,200,000  | 1816 | 30,573,000 | 1906 |   41,067,000    |
|  400  | 5,500,000  | 1821 | 31,578,000 | 1911 |   41,415,000    |
|  850  | 7,000,000  | 1826 | 32,665,000 | 1921 |   39,108,000    |
| 1226  | 16,000,000 | 1831 | 33,595,000 | 1926 |   40,581,000    |
| 1345  | 20,200,000 | 1836 | 34,293,000 | 1931 |   41,524,000    |
| 1400  | 16,600,000 | 1841 | 34,912,000 | 1936 |   41,502,000    |
| 1457  | 19,700,000 | 1846 | 36,097,000 | 1946 |   40,506,639    |
| 1580  | 20,000,000 | 1851 | 36,472,000 | 1954 |   42,777,162    |
| 1594  | 18,500,000 | 1856 | 36,715,000 | 1962 |   46,519,997    |
| 1600  | 20,000,000 | 1861 | 37,386,000 | 1968 |   49,780,543    |
| 1670  | 18,000,000 | 1866 | 38,067,000 | 1975 |   52,655,864    |
| 1700  | 21,000,000 | 1872 | 37,653,000 | 1982 |   54,334,871    |
| 1715  | 19,200,000 | 1876 | 38,438,000 | 1990 |   56,615,155    |
| 1740  | 24,600,000 | 1881 | 39,239,000 | 1999 |   58,520,688    |
| 1792  | 28,000,000 | 1886 | 39,783,000 | 2006 |   61,399,733    |
| 1801  | 29,361,000 | 1891 | 39,946,000 | 2016 |   64,513,000    |"""

# read the table; insert an "ignore" column bc reading the table yields an empty column 0.
df_population_fr = pd.read_table(StringIO(
            population_france_markdown.replace(",","").replace("hide ","")
        ), sep="|", skiprows=2, index_col=False,
        names=["ignore","Year0","Population0","Year1","Population1","Year2","Population2"])
# reindex the on a range of ints, and drop the unnecessary cols
df_population_fr["idx"] = range(0,df_population_fr.shape[0])
df_population_fr = df_population_fr.reindex(df_population_fr["idx"]).drop(["ignore","idx"],axis=1)
# reshape the 3 (year,pop)-pair horizontal sets of data into a single (year,pop) set of columns of data
df_population_fr = pd.lreshape(df_population_fr, {
    "year": ["Year0","Year1","Year2"], 
    "population": ["Population0","Population1","Population2"]})
# apparently France has kept records since the Roman Empire; rewrite the year column to be a negative number for BC years, centering 0 on 0AD.
df_population_fr["year"] = df_population_fr["year"].astype(str).str \
    .strip().replace(r"(\d+)\s*BC", r"-\1", regex=True).astype(int)
print(df_population_fr.dtypes)
# let's see the data starting from year of the first Fields Medal
df_population_fr[ df_population_fr["year"] >= 1936 ]

#%%
# calc avg French population from >= 1936
population_mean_fr = df_population_fr[ df_population_fr["year"] >= 1936 ]["population"].mean().astype(int)
print(f'mean resident_population (Census year >= 1936): {population_mean_fr}')

#%%[markdown]
### Comparison of France population values returned by 3 methods:
# - **ChatGPT: 52.1M** (see above)
#
# - **Wolfram Alpha: 49.99M**: [average population size france since 1936 - Wolfram|Alpha](https://www.wolframalpha.com/input?i=average+population+size+france+since+1936)
# 
# - **Our Census calculation: 51.74M**
#
#### So, we find: at-most a `abs(49.99/52.1 - 1) * 100 = 4.04%` difference between ChatGPT's calculation and ours.
### Is this "too large"? Again, probably not for this exercise.

#%%[markdown]
## Are we done verifying ChatGPT's data? It depends!
#
### If you want to be sure of data correctness, you MUST verify.
#### But as you can see, this is time-consuming.
#
### At-minimum, if factual correctness matters, then -- as in the pre-ChatGPT/large-language-model era -- you must spot-check ChatGPT (and any other LLM's) outputs, because at the time of this analysis. it contains no guarantees that prevent factual inaccuracies.
#
##### For now, we accept the US and France average population values as "within reason", and move-on.
### n.b. w.r.t. a career as a data analyst: ChatGPT's outputted aggregation values appear within tolerance for our purpose!

#%%[markdown]
## Having spot-checked and accepted the "average of historical data" values returned by ChatGPT, let's resume:

# %%
# copy-pasted text from ChatGPT
data_cgpt_raw = """Australia: 2 / 12.6 = 0.1587
Austria: 1 / 7.5 = 0.1333
Belgium: 2 / 9.9 = 0.2020
Brazil: 1 / 107.2 = 0.0093
Canada: 1 / 22.6 = 0.0442
China: 1 / 769.3 = 0.0000
Finland: 1 / 4.7 = 0.2128
France: 13 / 52.1 = 0.2494
Germany: 2 / 69.2 = 0.0289
Iran: 2 / 27.5 = 0.0727
Israel: 1 / 5.5 = 0.1818
Italy: 2 / 45.8 = 0.0437
Japan: 3 / 102.1 = 0.0294
New Zealand: 1 / 3.2 = 0.3125
Norway: 1 / 4.0 = 0.2500
Russia/Soviet Union: 9 / 134.8 = 0.0668
Sweden: 1 / 7.8 = 0.1282
Ukraine: 1 / 43.9 = 0.0228
United Kingdom: 8 / 56.5 = 0.1416
United States: 15 / 194.3 = 0.0773
Vietnam: 1 / 44.1 = 0.0227"""

# clean the input
data_cgpt = clean_chatgpt(data_cgpt_raw)
print(data_cgpt)   # cleaned CSV string, to spot-check QC


# %%[markdown]
### Uh-oh - ChatGPT uses '/' to denote synonyms for our merge of Russia and the Soviet Union.
#### Let's redefine `clean_chatgpt`, then re-run.
#
#### **Side-note about software engineering best-practice**:
# Rather than redefine `clean_chatgpt`, as we do below, it is **highly** preferable to update the original function to handle this case.
#
# This way the function always has exactly 1 definition, which prevents the SWE/DS/analyst from having to reason through the evolution of code at run-time.
#
# I'm breaking this rule for the pedadogy of illustrating that ChatGPT outputs aren't necessarily consistent, and that real-world data is messy.
#
# (Note: exceptions to this rule can exist for reasons of execution speed. But programming wisdom rightly advises that we first make our code work correctly, then optimize for performance.)

#%%
def clean_chatgpt(resp: str) -> str:
    return resp.replace("Russia/Soviet Union", "Russia & Soviet Union") \
        .replace(':', ',') \
        .replace('=', ',') \
        .replace('/',',') \
        .replace(" ", "")

# clean the input
data_cgpt = clean_chatgpt(data_cgpt_raw)
print(data_cgpt)   # cleaned CSV string, to spot-check QC

# read the CSV string into a DF
df = pd.read_csv(StringIO(data_cgpt), sep=',', names=["nat","fld","n","chatgpt_fld_per_megacapita"])
df

#%%
# do our own arithmetic, to the same 4 decimal places
df["our_fld_per_megacapita"] = df["fld"] / df["n"]
df["our_fld_per_megacapita_rounded"] = round(df["our_fld_per_megacapita"], 4)
df

#%%
# what's the (actual - expected) delta, as a % of ChatGPT's calculation?
df["chatgpt_error_pct"] = (df["chatgpt_fld_per_megacapita"] - df["our_fld_per_megacapita"]) / df["chatgpt_fld_per_megacapita"] * 100
df

# %%[markdown]
## Great! Looks like ChatGPT actually performed the arithmetic within rounding error.
### Now, how do the nations rank, in per-capita Fields Medal awards? Let's plot the result:

#%%
df.sort_values(by="our_fld_per_megacapita", ascending=False)[["nat","our_fld_per_megacapita"]].plot(
    x="nat", y="our_fld_per_megacapita", kind="bar", title="Field Medals per-capita (millions), by nation", 
    xlabel="Nation", ylabel="Field Medals per-capita (millions)", figsize=(10,5))

# %%[markdown]
## Analytical conclusion:
#### While the US has indeed acquired the most Fields Medals (22), when we contextualize that number by the population of each nation, it is clear that other nations - notably France (13) - are more prone to winning Fields Medals, per citizen headcount.
#### This is a very interesting result, as it shows that the US has indeed been able to acquire many Fields Medals, but that this success is not necessarily due to its superiority in mathematics research.
#
## ChatGPT usage conclusion:
#### We have seen that ChatGPT is capable of performing, in seconds, data-analytic tasks that previously required hours or days of work.
#### This has the potential to revolutionize the way we approach and solve problems, as it can provide us with valuable insights and solutions quickly and efficiently.
#### However, workers (such as data analysts and data journalists) whose value to an employer is predicated on producing such reports would be wise to consider their adaptation strategy to LLM tools such as ChatGPT.
#
#### Also, note: much of the code in this notebook is devoted to parsing ChatGPT output. **I could (and should) have asked ChatGPT to output results as CSV, further saving considerable time.**

# %%
