import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://www.scrapethissite.com/pages/simple/'
headers = {'User-Agent': 'Mozilla/5.0'}  
response = requests.get(url, headers=headers)

if response.status_code == 200:
    page_content = response.text
else:
    print(f'Failed to retrieve the page. Status code: {response.status_code}')

soup = BeautifulSoup(page_content, 'html.parser')

country_data = []

countries = soup.find_all('div', class_='col-md-4 country')

for country in countries:
    name_tag = country.find('h3', class_='country-name')
    name = name_tag.get_text(strip=True) if name_tag else 'N/A'
    
    capital_tag = country.find('span', class_='country-capital')
    capital = capital_tag.get_text(strip=True) if capital_tag else 'N/A'
    
    population_tag = country.find('span', class_='country-population')
    population = population_tag.get_text(strip=True) if population_tag else 'N/A'
    
    area_tag = country.find('span', class_='country-area')
    area = area_tag.get_text(strip=True) if area_tag else 'N/A'
    
    country_data.append({
        'Country': name,
        'Capital': capital,
        'Population': population,
        'Area (km²)': area
    })

df = pd.DataFrame(country_data)
print(df)

plt.figure(figsize=(8, 6))
plt.scatter(df['Area (km²)'], df['Population'], alpha=0.5, color='green')
plt.xlabel('Area (km²)')
plt.ylabel('Population')
plt.title('Area vs. Population')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

le_country = LabelEncoder()
le_capital = LabelEncoder()

df['Country'] = le_country.fit_transform(df['Country'])
df['Capital'] = le_capital.fit_transform(df['Capital'])


X = df.drop('Population', axis=1)
y = df['Population']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
