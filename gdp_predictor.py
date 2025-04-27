import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

try:
    print("Running using Polynomial Regression (No Deep Learning)...")

    years = np.array([
        1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005,
        2010, 2015, 2020, 2025
    ]).reshape(-1, 1)

    gdp = np.array([7, 9, 12, 18, 30, 35, 40, 55, 75, 110, 180, 230, 278, 375])
    population = np.array([45, 52, 60, 70, 80, 90, 100, 115, 130, 145, 170, 190, 220, 250])
    urbanization = np.array([22, 24, 26, 28, 30, 32, 35, 38, 40, 43, 47, 50, 55, 58])
    poverty = np.array([60, 58, 55, 50, 48, 45, 40, 35, 32, 28, 25, 20, 18, 15])

    poly = PolynomialFeatures(degree=3)
    years_poly = poly.fit_transform(years)

    model_gdp = LinearRegression()
    model_population = LinearRegression()
    model_urbanization = LinearRegression()
    model_poverty = LinearRegression()

    model_gdp.fit(years_poly, gdp)
    model_population.fit(years_poly, population)
    model_urbanization.fit(years_poly, urbanization)
    model_poverty.fit(years_poly, poverty)

    year_2070 = np.array([[2070]])
    year_2070_poly = poly.transform(year_2070)

    predicted_gdp_2070 = model_gdp.predict(year_2070_poly)[0]
    predicted_population_2070 = model_population.predict(year_2070_poly)[0]
    predicted_urbanization_2070 = model_urbanization.predict(year_2070_poly)[0]
    predicted_poverty_2070 = model_poverty.predict(year_2070_poly)[0]

    predicted_gdp_per_capita_2070 = (predicted_gdp_2070 * 1e9) / (predicted_population_2070 * 1e6)

    gdp_2025 = gdp[-1]
    population_2025 = population[-1]
    gdp_per_capita_2025 = (gdp_2025 * 1e9) / (population_2025 * 1e6)

    print(f"Pakistan's GDP in 2025: ${gdp_2025:.2f} billion")
    print(f"Pakistan's Population in 2025: {population_2025:.2f} million")
    print(f"Pakistan's GDP per capita in 2025: ${gdp_per_capita_2025:.2f}")
    print(f"\nPredicted Pakistan's GDP in 2070: ${predicted_gdp_2070:.2f} billion")
    print(f"Predicted Pakistan's Population in 2070: {predicted_population_2070:.2f} million")
    print(f"Predicted Pakistan's GDP per capita in 2070: ${predicted_gdp_per_capita_2070:.2f}")
    print(f"Predicted Urbanization Rate in 2070: {predicted_urbanization_2070:.2f}%")
    print(f"Predicted Poverty Rate in 2070: {predicted_poverty_2070:.2f}%")

    input("\nPress Enter to continue and view graphs...")

    future_years = np.arange(1960, 2073, 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)

    predicted_gdp_future = model_gdp.predict(future_years_poly)
    predicted_population_future = model_population.predict(future_years_poly)
    predicted_urbanization_future = model_urbanization.predict(future_years_poly)
    predicted_poverty_future = model_poverty.predict(future_years_poly)

    milestone_years = np.arange(2030, 2071, 5)
    milestone_years_poly = poly.transform(milestone_years.reshape(-1,1))

    milestone_gdp = model_gdp.predict(milestone_years_poly)
    milestone_population = model_population.predict(milestone_years_poly)
    milestone_urbanization = model_urbanization.predict(milestone_years_poly)
    milestone_poverty = model_poverty.predict(milestone_years_poly)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(years, gdp, color='blue', label='Historical GDP')
    plt.plot(future_years, predicted_gdp_future, color='green', label='Predicted GDP Curve')
    plt.scatter(milestone_years, milestone_gdp, color='red', marker='o', label='Milestone Predictions')
    plt.title("Pakistan GDP Prediction")
    plt.xlabel("Year")
    plt.ylabel("GDP (Billion USD)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(years, population, color='purple', label='Historical Population')
    plt.plot(future_years, predicted_population_future, color='orange', label='Predicted Population Curve')
    plt.scatter(milestone_years, milestone_population, color='red', marker='o', label='Milestone Predictions")
    plt.title("Pakistan Population Prediction")
    plt.xlabel("Year")
    plt.ylabel("Population (Millions)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(years, urbanization, color='teal', label='Historical Urbanization')
    plt.plot(future_years, predicted_urbanization_future, color='cyan', label='Predicted Urbanization Curve')
    plt.scatter(milestone_years, milestone_urbanization, color='red', marker='o', label='Milestone Predictions')
    plt.title("Pakistan Urbanization Prediction")
    plt.xlabel("Year")
    plt.ylabel("Urbanization (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(years, poverty, color='red', label='Historical Poverty')
    plt.plot(future_years, predicted_poverty_future, color='magenta', label='Predicted Poverty Curve')
    plt.scatter(milestone_years, milestone_poverty, color='blue', marker='o', label='Milestone Predictions')
    plt.title("Pakistan Poverty Prediction")
    plt.xlabel("Year")
    plt.ylabel("Poverty (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\n\n--- ERROR OCCURRED ---")
    print(e)
    input("\nPress Enter to exit...")