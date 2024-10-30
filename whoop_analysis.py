import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from datetime import timedelta

# Set a consistent style for all plots
plt.style.use('seaborn-whitegrid')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)


# Load data from CSV files
df_physio = pd.read_csv('physiological_cycles.csv')
df_workouts = pd.read_csv('workouts.csv')
df_sleep = pd.read_csv('sleep.csv')

# Custom function to parse dates
def parse_date(date_string):
    try:
        return parser.parse(date_string)
    except:
        return pd.NaT

# Convert date columns to datetime
date_columns = ['Cycle start time', 'Cycle end time', 'Sleep onset', 'Wake onset']
for col in date_columns:
    df_physio[col] = df_physio[col].apply(parse_date)

# Ensure workout and sleep data have datetime columns
df_workouts['start_time'] = df_workouts['start_time'].apply(parse_date)
df_sleep['start_time'] = df_sleep['start_time'].apply(parse_date)

# Remove rows with NaT values in date columns
df_physio = df_physio.dropna(subset=date_columns)
df_workouts = df_workouts.dropna(subset=['start_time'])
df_sleep = df_sleep.dropna(subset=['start_time'])

# Function to plot time series
def plot_time_series(x, y, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df_physio[x], df_physio[y], marker='o')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot scatter
def plot_scatter(x, y, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_physio, x=x, y=y)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()

# Example plots
plot_time_series('Cycle start time', 'Recovery score %', 'Recovery Score Over Time')
plot_time_series('Cycle start time', 'Resting heart rate (bpm)', 'Resting Heart Rate Over Time')
plot_scatter('Recovery score %', 'Resting heart rate (bpm)', 'Recovery Score vs Resting Heart Rate')

# Function to analyze sleep metrics
def analyze_sleep_metrics():
    plt.figure(figsize=(12, 6))
    plt.bar(df_physio['Cycle start time'], df_physio['Sleep performance %'])
    plt.title('Sleep Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sleep Performance %')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("Average Sleep Metrics:")
    print(f"Sleep Performance: {df_physio['Sleep performance %'].mean():.2f}%")
    print(f"Sleep Efficiency: {df_physio['Sleep efficiency %'].mean():.2f}%")
    print(f"Sleep Consistency: {df_physio['Sleep consistency %'].mean():.2f}%")

# Function to analyze strain and recovery
def analyze_strain_recovery():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_physio, x='Day Strain', y='Recovery score %')
    plt.title('Day Strain vs Recovery Score')
    plt.xlabel('Day Strain')
    plt.ylabel('Recovery Score %')
    plt.tight_layout()
    plt.show()

    correlation = df_physio['Day Strain'].corr(df_physio['Recovery score %'])
    print(f"Correlation between Day Strain and Recovery Score: {correlation:.2f}")

# Call the new functions
analyze_sleep_metrics()
analyze_strain_recovery()


# Convert workout date columns to datetime
workout_date_columns = ['Cycle start time', 'Cycle end time', 'Workout start time', 'Workout end time']
for col in workout_date_columns:
    df_workouts[col] = pd.to_datetime(df_workouts[col], errors='coerce')

# Remove rows with NaT values in date columns
df_workouts = df_workouts.dropna(subset=workout_date_columns)

# Function to analyze workout data
def analyze_workouts():
    # Activity distribution
    plt.figure(figsize=(12, 6))
    activity_counts = df_workouts['Activity name'].value_counts()
    activity_counts.plot(kind='bar')
    plt.title('Distribution of Workout Activities')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Average strain by activity
    plt.figure(figsize=(12, 6))
    avg_strain = df_workouts.groupby('Activity name')['Activity Strain'].mean().sort_values(ascending=False)
    avg_strain.plot(kind='bar')
    plt.title('Average Strain by Activity')
    plt.xlabel('Activity')
    plt.ylabel('Average Strain')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Workout duration vs. Strain
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_workouts, x='Duration (min)', y='Activity Strain')
    plt.title('Workout Duration vs. Strain')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Activity Strain')
    plt.tight_layout()
    plt.show()

    # Heart rate zone distribution
    hr_zones = ['HR Zone 1 %', 'HR Zone 2 %', 'HR Zone 3 %', 'HR Zone 4 %', 'HR Zone 5 %']
    avg_hr_zones = df_workouts[hr_zones].mean()
    plt.figure(figsize=(10, 6))
    avg_hr_zones.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Average Heart Rate Zone Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nWorkout Summary Statistics:")
    print(f"Total number of workouts: {len(df_workouts)}")
    print(f"Average workout duration: {df_workouts['Duration (min)'].mean():.2f} minutes")
    print(f"Average activity strain: {df_workouts['Activity Strain'].mean():.2f}")
    print(f"Average calories burned: {df_workouts['Energy burned (cal)'].mean():.2f}")

# Function to analyze recovery vs. workout intensity
def analyze_recovery_vs_workout():
    # Sort both dataframes by their respective time columns
    df_workouts_sorted = df_workouts.sort_values('Workout start time')
    df_physio_sorted = df_physio.sort_values('Cycle start time')

    # Merge workout data with physiological data
    merged_data = pd.merge_asof(df_workouts_sorted, 
                                df_physio_sorted[['Cycle start time', 'Recovery score %']], 
                                left_on='Workout start time', 
                                right_on='Cycle start time', 
                                direction='forward')

    # Remove rows where Recovery score % is NaN
    merged_data = merged_data.dropna(subset=['Recovery score %'])

    if len(merged_data) > 0:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_data, x='Activity Strain', y='Recovery score %')
        plt.title('Workout Strain vs. Next Day Recovery Score')
        plt.xlabel('Workout Strain')
        plt.ylabel('Next Day Recovery Score %')
        plt.tight_layout()
        plt.show()

        correlation = merged_data['Activity Strain'].corr(merged_data['Recovery score %'])
        print(f"\nCorrelation between Workout Strain and Next Day Recovery Score: {correlation:.2f}")
    else:
        print("No matching data found for recovery vs. workout analysis.")

# Call the new functions
analyze_workouts()
analyze_recovery_vs_workout()




# Convert sleep date columns to datetime
sleep_date_columns = ['Cycle start time', 'Cycle end time', 'Sleep onset', 'Wake onset']
for col in sleep_date_columns:
    df_sleep[col] = pd.to_datetime(df_sleep[col], errors='coerce')

# Remove rows with NaT values in date columns
df_sleep = df_sleep.dropna(subset=sleep_date_columns)

# Function to analyze sleep data
def analyze_sleep():
    # Sleep duration over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_sleep['Sleep onset'], df_sleep['Asleep duration (min)'] / 60, marker='o')
    plt.title('Sleep Duration Over Time')
    plt.xlabel('Sleep Onset')
    plt.ylabel('Sleep Duration (hours)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sleep stage distribution
    sleep_stages = ['Light sleep duration (min)', 'Deep (SWS) duration (min)', 'REM duration (min)']
    avg_sleep_stages = df_sleep[sleep_stages].mean()
    plt.figure(figsize=(10, 6))
    avg_sleep_stages.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Average Sleep Stage Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Sleep performance vs. respiratory rate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sleep, x='Respiratory rate (rpm)', y='Sleep performance %')
    plt.title('Sleep Performance vs. Respiratory Rate')
    plt.xlabel('Respiratory Rate (rpm)')
    plt.ylabel('Sleep Performance %')
    plt.tight_layout()
    plt.show()

    # Sleep efficiency over time
    plt.figure(figsize=(12, 6))
    plt.plot(df_sleep['Sleep onset'], df_sleep['Sleep efficiency %'], marker='o')
    plt.title('Sleep Efficiency Over Time')
    plt.xlabel('Sleep Onset')
    plt.ylabel('Sleep Efficiency %')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSleep Summary Statistics:")
    print(f"Average sleep duration: {df_sleep['Asleep duration (min)'].mean() / 60:.2f} hours")
    print(f"Average sleep efficiency: {df_sleep['Sleep efficiency %'].mean():.2f}%")
    print(f"Average sleep performance: {df_sleep['Sleep performance %'].mean():.2f}%")
    print(f"Average respiratory rate: {df_sleep['Respiratory rate (rpm)'].mean():.2f} rpm")

# Function to analyze sleep and recovery relationship
def analyze_sleep_recovery():
    # Merge sleep data with physiological data
    merged_data = pd.merge_asof(df_sleep.sort_values('Sleep onset'), 
                                df_physio[['Cycle start time', 'Recovery score %']].sort_values('Cycle start time'), 
                                left_on='Sleep onset', 
                                right_on='Cycle start time', 
                                direction='forward')

    # Remove rows where Recovery score % is NaN
    merged_data = merged_data.dropna(subset=['Recovery score %'])

    if len(merged_data) > 0:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_data, x='Sleep performance %', y='Recovery score %')
        plt.title('Sleep Performance vs. Next Day Recovery Score')
        plt.xlabel('Sleep Performance %')
        plt.ylabel('Next Day Recovery Score %')
        plt.tight_layout()
        plt.show()

        correlation = merged_data['Sleep performance %'].corr(merged_data['Recovery score %'])
        print(f"\nCorrelation between Sleep Performance and Next Day Recovery Score: {correlation:.2f}")
    else:
        print("No matching data found for sleep and recovery analysis.")

# Call the new functions
analyze_sleep()
analyze_sleep_recovery()

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import seasonal_decompose


def perform_cluster_analysis():
    # Select relevant features for clustering
    features = ['Recovery score %', 'Resting heart rate (bpm)', 'Heart rate variability (ms)', 
                'Sleep performance %', 'Sleep efficiency %', 'Day Strain']
    
    # Create a pipeline for imputation, normalization, and clustering
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])
    
    # Fit the pipeline and predict clusters
    df_physio['Cluster'] = pipeline.fit_predict(df_physio[features])
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_physio, x='Recovery score %', y='Day Strain', hue='Cluster', palette='viridis')
    plt.title('Clusters of Recovery Score vs Day Strain')
    plt.show()
    
    # Analyze cluster characteristics
    cluster_means = df_physio.groupby('Cluster')[features].mean()
    print("Cluster Characteristics:")
    print(cluster_means)
    
    # Visualize cluster characteristics
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Cluster Characteristics Heatmap')
    plt.show()

    # Improved cluster visualization
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(data=df_physio, x='Recovery score %', y='Day Strain', 
                              hue='Cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters of Recovery Score vs Day Strain', fontsize=16)
    plt.xlabel('Recovery Score (%)', fontsize=12)
    plt.ylabel('Day Strain', fontsize=12)
    plt.legend(title='Cluster', title_fontsize='12', fontsize='10')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Improved heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
    plt.title('Cluster Characteristics Heatmap', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()



def build_recovery_prediction_model():
    # Prepare features and target
    features = ['Day Strain', 'Sleep performance %', 'Sleep efficiency %', 'Resting heart rate (bpm)', 
                'Heart rate variability (ms)', 'Respiratory rate (rpm)']
    X = df_physio[features]
    y = df_physio['Recovery score %']
    
    # Remove rows where the target variable (Recovery score %) is NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline that includes imputation, scaling, and the Random Forest model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': pipeline.named_steps['rf'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Recovery Score Prediction')
    plt.show()

    return pipeline  # Return the trained pipeline for potential future use

# Run the function
recovery_model = build_recovery_prediction_model()


def perform_time_series_analysis():
    # Ensure 'Cycle start time' is a datetime
    df_physio['Cycle start time'] = pd.to_datetime(df_physio['Cycle start time'], errors='coerce')
    
    # Set 'Cycle start time' as the index and sort
    df_physio_sorted = df_physio.set_index('Cycle start time').sort_index()
    
    # Resample data to daily frequency and calculate mean recovery score
    daily_recovery = df_physio_sorted['Recovery score %'].resample('D').mean()
    
    # Interpolate missing values
    daily_recovery_interpolated = daily_recovery.interpolate()
    
    # Ensure the time series has no gaps
    daily_recovery_continuous = daily_recovery_interpolated.asfreq('D')
    
    # If there are still NaN values at the beginning or end, drop them
    daily_recovery_clean = daily_recovery_continuous.dropna()
    
    if len(daily_recovery_clean) < 14:  # Need at least two periods for decomposition
        print("Not enough data for seasonal decomposition after cleaning.")
        return
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(daily_recovery_clean, model='additive', period=7)
    
    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nTime Series Analysis:")
    print(f"Mean Recovery Score: {daily_recovery_clean.mean():.2f}")
    print(f"Standard Deviation of Recovery Score: {daily_recovery_clean.std():.2f}")
    print(f"Minimum Recovery Score: {daily_recovery_clean.min():.2f}")
    print(f"Maximum Recovery Score: {daily_recovery_clean.max():.2f}")

    # Improved time series decomposition plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20))
    decomposition.observed.plot(ax=ax1, color='#1f77b4')
    ax1.set_title('Observed', fontsize=14)
    decomposition.trend.plot(ax=ax2, color='#ff7f0e')
    ax2.set_title('Trend', fontsize=14)
    decomposition.seasonal.plot(ax=ax3, color='#2ca02c')
    ax3.set_title('Seasonal', fontsize=14)
    decomposition.resid.plot(ax=ax4, color='#d62728')
    ax4.set_title('Residual', fontsize=14)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('')
    
    fig.suptitle('Time Series Decomposition of Recovery Score', fontsize=16)
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()



def detect_anomalies():
    # Select features for anomaly detection
    features = ['Recovery score %', 'Day Strain', 'Sleep performance %', 'Resting heart rate (bpm)']
    
    # Create a pipeline for imputation, normalization, and anomaly detection
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('iso_forest', IsolationForest(contamination=0.1, random_state=42))
    ])
    
    # Fit the pipeline and predict anomalies
    df_physio['Anomaly'] = pipeline.fit_predict(df_physio[features])
    
    # Visualize anomalies
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_physio, x='Day Strain', y='Recovery score %', hue='Anomaly', palette={1: 'blue', -1: 'red'})
    plt.title('Anomaly Detection: Day Strain vs Recovery Score')
    plt.show()
    
    # Analyze anomalies
    anomalies = df_physio[df_physio['Anomaly'] == -1]
    print("Anomaly Statistics:")
    print(anomalies[features].describe())

    # Improved anomaly detection plot
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(data=df_physio, x='Day Strain', y='Recovery score %', 
                              hue='Anomaly', palette={1: '#1f77b4', -1: '#d62728'}, 
                              s=100, alpha=0.7)
    plt.title('Anomaly Detection: Day Strain vs Recovery Score', fontsize=16)
    plt.xlabel('Day Strain', fontsize=12)
    plt.ylabel('Recovery Score (%)', fontsize=12)
    plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'], title_fontsize='12', fontsize='10')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run advanced analyses
perform_cluster_analysis()
build_recovery_prediction_model()
perform_time_series_analysis()
detect_anomalies()


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np


def build_advanced_recovery_model():
    # Prepare features and target
    features = ['Day Strain', 'Sleep performance %', 'Sleep efficiency %', 'Resting heart rate (bpm)', 
                'Heart rate variability (ms)', 'Respiratory rate (rpm)']
    X = df_physio[features]
    y = df_physio['Recovery score %']
    
    # Remove rows where the target variable (Recovery score %) is NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Create a pipeline for imputation, scaling, and model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameter search space
    param_dist = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
    
    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                       n_iter=20, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    
    # Print best parameters and score
    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", random_search.best_score_)
    
    # Evaluate model using cross-validation
    cv_scores = cross_val_score(random_search.best_estimator_, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': random_search.best_estimator_.named_steps['rf'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Improved feature importance plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette="viridis")
    plt.title('Feature Importance for Recovery Score Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return random_search.best_estimator_

# Run the function
advanced_recovery_model = build_advanced_recovery_model()


def generate_recommendations(user_data):
    recommendations = []
    
    # Recovery score recommendation
    if user_data['Recovery score %'] < 33:
        recommendations.append("Your recovery is low. Consider taking a rest day or doing light activity.")
    elif user_data['Recovery score %'] < 66:
        recommendations.append("Your recovery is moderate. Focus on maintaining good sleep habits and manage your strain.")
    else:
        recommendations.append("Your recovery is high. You're ready for high-intensity training if desired.")
    
    # Sleep recommendations
    if user_data['Sleep performance %'] < 70:
        recommendations.append("Your sleep performance is low. Try to improve your sleep habits and aim for more consistent sleep times.")
    
    if user_data['Sleep efficiency %'] < 85:
        recommendations.append("Your sleep efficiency could be improved. Minimize disturbances and create a better sleep environment.")
    
    # Strain recommendations
    if user_data['Day Strain'] > 15 and user_data['Recovery score %'] < 50:
        recommendations.append("Your strain is high relative to your recovery. Consider reducing your training intensity.")
    
    # HRV recommendations
    if user_data['Heart rate variability (ms)'] < user_data['Heart rate variability (ms)'].mean():
        recommendations.append("Your HRV is below your average. This might indicate accumulated stress or fatigue. Focus on recovery activities.")
    
    return recommendations

# Example usage
recent_data = df_physio.iloc[-1]  # Get the most recent data point
user_recommendations = generate_recommendations(recent_data)

print("Personalized Recommendations:")
for rec in user_recommendations:
    print("- " + rec)





def get_user_input():
    print("Please enter your data for today:")
    user_input = {}
    user_input['Sleep performance %'] = float(input("Sleep performance %: "))
    user_input['Heart rate variability (ms)'] = float(input("Heart rate variability (ms): "))
    user_input['Resting heart rate (bpm)'] = float(input("Resting heart rate (bpm): "))
    user_input['Respiratory rate (rpm)'] = float(input("Respiratory rate (rpm): "))
    user_input['Recovery score %'] = float(input("Recovery score %: "))
    return user_input

def analyze_and_recommend(user_input):
    # Prepare the input data for the model
    features = ['Day Strain', 'Sleep performance %', 'Sleep efficiency %', 'Resting heart rate (bpm)', 
                'Heart rate variability (ms)', 'Respiratory rate (rpm)']
    
    # Create a DataFrame with the user input
    user_data = pd.DataFrame([user_input])
    
    # Handle missing or extra features
    if 'Day Strain' not in user_data:
        user_data['Day Strain'] = df_physio['Day Strain'].median()  # Use median as a placeholder
    if 'Sleep efficiency %' not in user_data:
        user_data['Sleep efficiency %'] = user_data['Sleep performance %']  # Use sleep performance as a proxy
    if 'Restorative sleep %' in user_data:
        user_data = user_data.drop(columns=['Restorative sleep %'])  # Remove this if it's not used in the model
    
    # Make sure all required features are present
    for feature in features:
        if feature not in user_data.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Use the trained model to predict Recovery score
    predicted_recovery = advanced_recovery_model.predict(user_data[features])[0]
    
    # Generate recommendations
    recommendations = generate_recommendations(user_data.iloc[0])
    
    # Prepare the results
    results = {
        'Actual Recovery Score': user_input['Recovery score %'],
        'Predicted Recovery Score': round(predicted_recovery, 2),
        'Recommendations': recommendations,
        'Analysis': []
    }
    
    # Add some analysis based on the input and prediction
    if predicted_recovery > user_input['Recovery score %']:
        results['Analysis'].append("Your predicted recovery score is higher than your actual recovery. This suggests potential for improvement in your recovery state.")
    else:
        results['Analysis'].append("Your predicted recovery score is in line with or lower than your actual recovery. Focus on maintaining your current recovery practices.")
    
    if user_input['Sleep performance %'] < 80:
        results['Analysis'].append("Your sleep performance is below optimal levels. Improving sleep quality could significantly boost your recovery.")
    
    if user_input['Heart rate variability (ms)'] < 50:  # This threshold might need adjustment based on your typical HRV
        results['Analysis'].append("Your HRV is relatively low, which might indicate accumulated stress or fatigue. Consider focusing on recovery activities.")
    
    return results
    
   

def generate_recommendations(user_data):
    recommendations = []
    
    if user_data['Recovery score %'] < 33:
        recommendations.append("Your recovery is low. Consider taking a rest day or doing light activity.")
    elif user_data['Recovery score %'] < 66:
        recommendations.append("Your recovery is moderate. Focus on maintaining good sleep habits and manage your strain.")
    else:
        recommendations.append("Your recovery is high. You're ready for high-intensity training if desired.")
    
    if user_data['Sleep performance %'] < 70:
        recommendations.append("Your sleep performance is low. Try to improve your sleep habits and aim for more consistent sleep times.")
    
    if user_data['Heart rate variability (ms)'] < user_data['Heart rate variability (ms)'].mean():
        recommendations.append("Your HRV is below your average. This might indicate accumulated stress or fatigue. Focus on recovery activities.")
    
    return recommendations

# Assume the advanced_recovery_model is already trained and available
def predict_tomorrow_metrics(user_input, strain_levels=[5, 10, 15, 20]):
    features = ['Day Strain', 'Sleep performance %', 'Sleep efficiency %', 'Resting heart rate (bpm)', 
                'Heart rate variability (ms)', 'Respiratory rate (rpm)']
    
    predictions = []
    
    for strain in strain_levels:
        # Create a copy of user input and add the strain level
        tomorrow_data = user_input.copy()
        tomorrow_data['Day Strain'] = strain
        
        # Use today's sleep performance as a proxy for tomorrow's sleep efficiency
        if 'Sleep efficiency %' not in tomorrow_data:
            tomorrow_data['Sleep efficiency %'] = tomorrow_data['Sleep performance %']
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([tomorrow_data])
        
        # Ensure all required features are present
        for feature in features:
            if feature not in input_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Use the model to predict tomorrow's recovery score
        predicted_recovery = advanced_recovery_model.predict(input_df[features])[0]
        
        # You might want to add more sophisticated predictions for other metrics here
        # For now, we'll use simple rules based on strain
        predicted_hrv = max(user_input['Heart rate variability (ms)'] - strain/2, 20)
        predicted_rhr = min(user_input['Resting heart rate (bpm)'] + strain/3, 100)
        predicted_respiratory_rate = min(user_input['Respiratory rate (rpm)'] + strain/10, 20)
        
        predictions.append({
            'Day Strain': strain,
            'Predicted Recovery Score': round(predicted_recovery, 2),
            'Predicted HRV': round(predicted_hrv, 2),
            'Predicted RHR': round(predicted_rhr, 2),
            'Predicted Respiratory Rate': round(predicted_respiratory_rate, 2)
        })
    
    # Visualize predictions
    plt.figure(figsize=(14, 10))
    metrics = ['Predicted Recovery Score', 'Predicted HRV', 'Predicted RHR', 'Predicted Respiratory Rate']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.lineplot(x=[p['Day Strain'] for p in predictions], 
                     y=[p[metric] for p in predictions], 
                     marker='o', color=colors[i])
        plt.title(metric, fontsize=14)
        plt.xlabel('Day Strain', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('tomorrow_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions

# Main execution
if __name__ == "__main__":
    user_input = get_user_input()
    analysis_results = analyze_and_recommend(user_input)

    print("\nAnalysis and Recommendations:")
    print(f"Actual Recovery Score: {analysis_results['Actual Recovery Score']}%")
    print(f"Predicted Recovery Score: {analysis_results['Predicted Recovery Score']}%")
    print("\nRecommendations:")
    for rec in analysis_results['Recommendations']:
        print(f"- {rec}")
    print("\nAnalysis:")
    for analysis in analysis_results['Analysis']:
        print(f"- {analysis}")
    
    # Predict tomorrow's metrics
    tomorrow_predictions = predict_tomorrow_metrics(user_input)
    
    print("\nPredicted Metrics for Tomorrow based on Different Strain Levels:")
    for pred in tomorrow_predictions:
        print(f"\nIf Day Strain is {pred['Day Strain']}:")
        print(f"  Predicted Recovery Score: {pred['Predicted Recovery Score']}%")
        print(f"  Predicted HRV: {pred['Predicted HRV']} ms")
        print(f"  Predicted RHR: {pred['Predicted RHR']} bpm")
        print(f"  Predicted Respiratory Rate: {pred['Predicted Respiratory Rate']} rpm")


# Load Hevy data
def load_and_preprocess_hevy_data():
    # Load Hevy dataset
    df_hevy = pd.read_csv('workout_data_hevy.csv')
    
    # Convert time columns to datetime
    df_hevy['start_time'] = pd.to_datetime(df_hevy['start_time'])
    df_hevy['end_time'] = pd.to_datetime(df_hevy['end_time'])
    
    # Calculate workout duration in minutes
    df_hevy['workout_duration'] = (df_hevy['end_time'] - df_hevy['start_time']).dt.total_seconds() / 60
    
    # Group by workout (title and start_time) to get workout-level metrics
    workout_summary = df_hevy.groupby(['title', 'start_time']).agg({
        'end_time': 'first',
        'exercise_title': 'count',
        'weight_kg': 'sum',
        'reps': 'sum',
        'distance_km': 'sum',
        'duration_seconds': 'sum',
        'rpe': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    workout_summary = workout_summary.rename(columns={
        'exercise_title': 'total_exercises',
        'weight_kg': 'total_weight_kg',
        'reps': 'total_reps',
        'distance_km': 'total_distance_km',
        'duration_seconds': 'total_duration_seconds',
        'rpe': 'avg_rpe'
    })
    
    return df_hevy, workout_summary

def analyze_hevy_workouts(df_hevy, workout_summary):
    # Workout frequency analysis
    plt.figure(figsize=(12, 6))
    workout_summary['start_time'].dt.date.value_counts().sort_index().plot(kind='line', marker='o')
    plt.title('Workout Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Workouts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Exercise type distribution
    plt.figure(figsize=(12, 6))
    df_hevy['exercise_title'].value_counts().head(15).plot(kind='bar')
    plt.title('Most Common Exercises')
    plt.xlabel('Exercise')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # RPE distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_hevy, x='rpe', bins=10)
    plt.title('RPE Distribution')
    plt.xlabel('RPE')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Volume progression over time
    volume_over_time = workout_summary.groupby(workout_summary['start_time'].dt.date).agg({
        'total_weight_kg': 'sum',
        'total_reps': 'sum'
    }).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(volume_over_time['start_time'], volume_over_time['total_weight_kg'], marker='o')
    ax1.set_title('Total Weight Lifted Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Weight (kg)')
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(volume_over_time['start_time'], volume_over_time['total_reps'], marker='o', color='orange')
    ax2.set_title('Total Reps Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Reps')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def combine_whoop_and_hevy_analysis(df_physio, workout_summary):
    # Merge Whoop and Hevy data based on dates
    merged_data = pd.merge_asof(
        workout_summary.sort_values('start_time'),
        df_physio[['Cycle start time', 'Recovery score %', 'Day Strain']].sort_values('Cycle start time'),
        left_on='start_time',
        right_on='Cycle start time',
        direction='nearest',
        tolerance=pd.Timedelta('1d')
    )

    # Analyze relationship between Recovery Score and workout metrics
    plt.figure(figsize=(15, 10))
    
    # Recovery Score vs RPE
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=merged_data, x='Recovery score %', y='avg_rpe')
    plt.title('Recovery Score vs Average RPE')
    
    # Recovery Score vs Total Volume
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=merged_data, x='Recovery score %', y='total_weight_kg')
    plt.title('Recovery Score vs Total Weight Lifted')
    
    # Day Strain vs RPE
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=merged_data, x='Day Strain', y='avg_rpe')
    plt.title('Day Strain vs Average RPE')
    
    # Day Strain vs Total Volume
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=merged_data, x='Day Strain', y='total_weight_kg')
    plt.title('Day Strain vs Total Weight Lifted')
    
    plt.tight_layout()
    plt.show()

    # Calculate correlations
    correlations = {
        'Recovery Score vs RPE': merged_data['Recovery score %'].corr(merged_data['avg_rpe']),
        'Recovery Score vs Total Weight': merged_data['Recovery score %'].corr(merged_data['total_weight_kg']),
        'Day Strain vs RPE': merged_data['Day Strain'].corr(merged_data['avg_rpe']),
        'Day Strain vs Total Weight': merged_data['Day Strain'].corr(merged_data['total_weight_kg'])
    }

    print("\nCorrelations:")
    for key, value in correlations.items():
        print(f"{key}: {value:.3f}")

    return merged_data

# Load and analyze Hevy data
df_hevy, workout_summary = load_and_preprocess_hevy_data()
analyze_hevy_workouts(df_hevy, workout_summary)

# Combine Whoop and Hevy analysis
merged_analysis = combine_whoop_and_hevy_analysis(df_physio, workout_summary)


## Spyder Plots - Goal Vs current Scores
def create_strength_progression_plot(df_hevy, body_weight=85):
    # Define target goals based on body weight multipliers
    targets = {
        'Squat': body_weight * 2,      # 2x bodyweight
        'Bench Press': body_weight * 1.5,  # 1.5x bodyweight
        'Row': body_weight * 2,        # 2x bodyweight
        'Deadlift': body_weight * 2.5,    # 2.5x bodyweight
        'Power Clean': body_weight * 1.5   # 1.5x bodyweight
    }
    
    # Get max weights for each exercise over time
    exercise_progress = {}
    for exercise in targets.keys():
        # Create a list of possible variations of the exercise name
        exercise_variations = [
            exercise.lower(),
            exercise.replace(' ', '').lower(),
            f"{exercise.lower()} (barbell)",
            f"barbell {exercise.lower()}"
        ]
        
        # Filter for the specific exercise and its variations
        exercise_data = df_hevy[
            df_hevy['exercise_title'].str.lower().str.contains('|'.join(exercise_variations), na=False)
        ]
        
        if not exercise_data.empty:
            exercise_progress[exercise] = exercise_data.groupby(
                exercise_data['start_time'].dt.date)['weight_kg'].max()
    
    if not exercise_progress:
        print("No matching exercises found in the dataset.")
        return
    
    # Get all unique dates
    all_dates = sorted(list(set.union(*[set(prog.index) for prog in exercise_progress.values()])))
    
    if len(all_dates) < 2:
        print("Not enough data points for progression analysis.")
        return
    
    # Calculate time points ensuring we don't exceed the list length
    n_points = 6  # 0%, 20%, 40%, 60%, 80%, 100%
    time_points = []
    for i in range(n_points):
        idx = int((i * (len(all_dates) - 1)) / (n_points - 1))
        time_points.append(all_dates[idx])
    
    # Get the max weights at each time point
    progression_data = []
    for date in time_points:
        point_data = {}
        for exercise in targets.keys():
            if exercise in exercise_progress:
                # Get the latest max weight up to this date
                weights_up_to_date = exercise_progress[exercise][exercise_progress[exercise].index <= date]
                if not weights_up_to_date.empty:
                    point_data[exercise] = weights_up_to_date.max()
                else:
                    point_data[exercise] = 0
            else:
                point_data[exercise] = 0
        progression_data.append(point_data)
    
    # Create the spider plot
    categories = list(targets.keys())
    n_categories = len(categories)
    
    # Set up the angles for the spider plot
    angles = [n/float(n_categories)*2*np.pi for n in range(n_categories)]
    angles += angles[:1]  # complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
    
    # Plot target values
    target_values = [targets[cat] for cat in categories]
    target_values += target_values[:1]
    ax.plot(angles, target_values, 'o-', linewidth=2, label='Target', color='red')
    ax.fill(angles, target_values, alpha=0.25, color='red')
    
    # Colors for progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(progression_data)))
    
    # Plot progression data
    for i, data in enumerate(progression_data):
        values = [data[cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1, label=f'Progress {i*20}%', color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(f'Strength Progression Towards Goals (BW: {body_weight}kg)', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print progression details
    print("\nProgression Details:")
    print("Exercise      | Target (kg) | Current (kg) | % of Target")
    print("-" * 55)
    current_data = progression_data[-1]
    for exercise in categories:
        target = targets[exercise]
        current = current_data[exercise]
        percentage = (current / target) * 100
        print(f"{exercise:<12} | {target:>10.1f} | {current:>11.1f} | {percentage:>10.1f}%")
    
    # Print exercise variations found
    print("\nExercise names found in dataset:")
    for exercise in targets.keys():
        variations = df_hevy[df_hevy['exercise_title'].str.contains(exercise, case=False, na=False)]['exercise_title'].unique()
        if len(variations) > 0:
            print(f"\n{exercise}:")
            for var in variations:
                print(f"  - {var}")
        else:
            print(f"\n{exercise}: No matching exercises found")

# Call the function
create_strength_progression_plot(df_hevy, body_weight=85)


