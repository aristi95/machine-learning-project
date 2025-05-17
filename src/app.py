from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

# Convert categorical variables to numerical
df["sex"] = df["sex"].replace({"female": 1, "male": 0})
df["smoker"] = df["smoker"].replace({"yes": 1, "no": 0})
df["region"] = df["region"].replace({"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4})

# Correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize relationships with charges
fig, axis = plt.subplots(2, 3, figsize=(15, 10))
sns.regplot(ax=axis[0, 0], data=df, x="age", y="charges")
sns.heatmap(df[["charges", "age"]].corr(), annot=True, fmt=".2f", ax=axis[1, 0], cbar=False)
sns.regplot(ax=axis[0, 1], data=df, x="bmi", y="charges")
sns.heatmap(df[["charges", "bmi"]].corr(), annot=True, fmt=".2f", ax=axis[1, 1], cbar=False)
sns.boxplot(ax=axis[0, 2], data=df, x="smoker", y="charges")
sns.heatmap(df[["charges", "smoker"]].corr(), annot=True, fmt=".2f", ax=axis[1, 2], cbar=False)
plt.suptitle('Relationships with Charges (Top: Plots, Bottom: Correlations)')
plt.tight_layout()
plt.show()

# Remove weak predictors
df.drop(columns=['sex', 'children', 'region'], inplace=True)

# Feature scaling and transformation
scaler = StandardScaler()
df[['bmi_scaled', 'age_scaled']] = scaler.fit_transform(df[['bmi', 'age']])
df['log_charges'] = np.log1p(df['charges'])
df = df[['age_scaled', 'bmi_scaled', 'smoker', 'log_charges']]

# Split data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train = df_train.drop('log_charges', axis=1).reset_index(drop=True)
y_train = df_train['log_charges'].reset_index(drop=True)
X_test = df_test.drop('log_charges', axis=1).reset_index(drop=True)
y_test = df_test['log_charges'].reset_index(drop=True)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print(f"Intercept (a): {model.intercept_}")
print(f"Coefficients (b): {model.coef_}")

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Visualization of results
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# Age vs log_charges
regression_equation_age = lambda x: model.intercept_ + model.coef_[0] * x
sns.scatterplot(ax=axes[0, 0], data=df_test, x="age_scaled", y="log_charges")
sns.lineplot(ax=axes[0, 0], x=df_test["age_scaled"], y=regression_equation_age(df_test["age_scaled"]))
axes[0, 0].set_title('Actual: Age vs Log Charges')
sns.scatterplot(ax=axes[0, 1], x=df_test["age_scaled"], y=y_pred)
sns.lineplot(ax=axes[0, 1], x=df_test["age_scaled"], y=regression_equation_age(df_test["age_scaled"]))
axes[0, 1].set_title('Predicted: Age vs Log Charges')

# BMI vs log_charges
regression_equation_bmi = lambda x: model.intercept_ + model.coef_[1] * x
sns.regplot(ax=axes[1, 0], data=df_test, x="bmi_scaled", y="log_charges")
sns.lineplot(ax=axes[1, 0], x=df_test["bmi_scaled"], y=regression_equation_bmi(df_test["bmi_scaled"]))
axes[1, 0].set_title('Actual: BMI vs Log Charges')
sns.regplot(ax=axes[1, 1], x=df_test["bmi_scaled"], y=y_pred)
sns.lineplot(ax=axes[1, 1], x=df_test["bmi_scaled"], y=regression_equation_bmi(df_test["bmi_scaled"]))
axes[1, 1].set_title('Predicted: BMI vs Log Charges')

# Smoker vs log_charges
regression_equation_smoker = lambda x: model.intercept_ + model.coef_[2] * x
sns.boxplot(ax=axes[2, 0], data=df_test, x="smoker", y="log_charges")
sns.lineplot(ax=axes[2, 0], x=df_test["smoker"], y=regression_equation_smoker(df_test["smoker"]))
axes[2, 0].set_title('Actual: Smoker vs Log Charges')
sns.boxplot(ax=axes[2, 1], x=df_test["smoker"], y=y_pred)
sns.lineplot(ax=axes[2, 1], x=df_test["smoker"], y=regression_equation_smoker(df_test["smoker"]))
axes[2, 1].set_title('Predicted: Smoker vs Log Charges')

plt.tight_layout()
plt.show()