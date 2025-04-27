# Pakistan GDP and Demographic Predictor (1960–2070)

This project predicts Pakistan's GDP, Population, Urbanization, and Poverty Rates from the year 1960 to 2070.

It uses **Polynomial Regression (Degree 3)** to fit historical data and forecast future trends based on realistic growth patterns.
The program also highlights important milestones (2030, 2035, 2040, etc.) on beautifully plotted graphs.

## 📈 Features

- Predicts **GDP** (in Billion USD)
- Predicts **Population** (in Millions)
- Predicts **GDP per Capita**
- Predicts **Urbanization Rate** (% living in cities)
- Predicts **Poverty Rate** (% below poverty line)
- Smooth year-by-year predictions till 2070
- Highlights predictions for every 5 years
- Fully lightweight (no GPU or Deep Learning required)

## 🛠 How to Install

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Pakistan-GDP-Predictor.git
cd Pakistan-GDP-Predictor
```

2. Install required libraries:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

```bash
python gdp_predictor.py
```

## 📜 Libraries Used

- numpy
- matplotlib
- scikit-learn

All libraries are listed in `requirements.txt`.

## 🤝 Contributing

This project is open for improvements!
If you have better models, larger datasets, or optimization ideas, feel free to fork and submit a pull request.

---

© 2025 Pakistan GDP Predictor | Licensed under MIT License
