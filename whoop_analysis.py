"""
Whoop and Hevy Data Analysis

This script analyzes fitness data from Whoop (recovery/strain metrics) and 
Hevy (workout tracking) to provide insights into training patterns, recovery 
relationships, and overall fitness trends.

Features:
- Recovery and strain analysis from Whoop data
- Workout performance analysis from Hevy data
- Combined analysis of recovery vs. workout performance
- Strength progression tracking
- Machine learning-based predictions and anomaly detection

Author: Caio Fonseca
Date: 2024-10-30
"""

#######################
# Import Dependencies #
#######################

# Data manipulation and analysis
import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
from dateutil import parser

# Data visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and statistical analysis
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import seasonal_decompose

#########################
# Configuration Settings #
#########################

# Set random seed for reproducibility
np.random.seed(42)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Configure matplotlib settings for better visualization
plt.style.use('seaborn-whitegrid')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

############################
# Data Analysis Parameters #
############################

# Define key metrics for analysis
METRICS = {
    'recovery': 'Recovery score %',
    'strain': 'Day Strain', 
    'hrv': 'Heart rate variability (ms)',
    'rhr': 'Resting heart rate (bpm)',
    'respiratory_rate': 'Respiratory rate (rpm)'
}

# Define thresholds for analysis
THRESHOLDS = {
    'low_recovery': 33,
    'medium_recovery': 66,
    'high_recovery': 100,
    'sleep_performance': 70
}

# Define strength targets (multipliers of bodyweight)
STRENGTH_TARGETS = {
    'Squat': 2.0,
    'Bench Press': 1.5,
    'Row': 2.0,
    'Deadlift': 2.5,
    'Power Clean': 1.5
}

#########################
# Data Loading Functions #
#########################

def load_whoop_data():
    """
    Load and preprocess Whoop data from CSV files
    
    Returns:
        tuple: Preprocessed dataframes (physiological, workouts, sleep)
    """
    # Load data
    df_physio = pd.read_csv('physiological_cycles.csv')
    df_workouts = pd.read_csv('workouts.csv')
    df_sleep = pd.read_csv('sleep.csv')
    
    # Convert date columns
    date_columns = ['Cycle start time', 'Cycle end time', 'Sleep onset', 'Wake onset']
    df_physio = clean_date_columns(df_physio, date_columns)
    df_workouts = clean_date_columns(df_workouts, ['start_time'])
    df_sleep = clean_date_columns(df_sleep, ['start_time'])
    
    return df_physio, df_workouts, df_sleep

def load_hevy_data():
    """
    Load and preprocess Hevy workout data
    
    Returns:
        tuple: Raw and summarized workout data
    """
    # Load data
    df_hevy = pd.read_csv('workout_data_hevy.csv')
    
    # Convert time columns to datetime
    df_hevy['start_time'] = pd.to_datetime(df_hevy['start_time'])
    df_hevy['end_time'] = pd.to_datetime(df_hevy['end_time'])
    
    # Calculate workout duration
    df_hevy['workout_duration'] = (df_hevy['end_time'] - df_hevy['start_time']).dt.total_seconds() / 60
    
    # Create workout summary
    workout_summary = summarize_workouts(df_hevy)
    
    return df_hevy, workout_summary

#########################
# Data Helper Functions #
#########################

def clean_date_columns(df, date_columns):
    """
    Convert date columns to datetime and handle missing values
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df.dropna(subset=date_columns)

def summarize_workouts(df):
    """
    Create workout-level summary statistics
    """
    return df.groupby(['title', 'start_time']).agg({
        'end_time': 'first',
        'exercise_title': 'count',
        'weight_kg': 'sum',
        'reps': 'sum',
        'distance_km': 'sum',
        'duration_seconds': 'sum',
        'rpe': 'mean'
    }).reset_index().rename(columns={
        'exercise_title': 'total_exercises',
        'weight_kg': 'total_weight_kg',
        'reps': 'total_reps',
        'distance_km': 'total_distance_km',
        'duration_seconds': 'total_duration_seconds',
        'rpe': 'avg_rpe'
    })

#############################
# Whoop Analysis Functions #
#############################

def analyze_recovery_patterns(df):
    """
    Analyze recovery score patterns and trends
    """
    recovery_stats = {
        'mean_recovery': df[METRICS['recovery']].mean(),
        'std_recovery': df[METRICS['recovery']].std(),
        'min_recovery': df[METRICS['recovery']].min(),
        'max_recovery': df[METRICS['recovery']].max()
    }
    
    df['recovery_trend'] = df[METRICS['recovery']].rolling(window=7).mean()
    
    recovery_consistency = {
        'days_above_66': (df[METRICS['recovery']] > THRESHOLDS['medium_recovery']).mean() * 100,
        'days_below_33': (df[METRICS['recovery']] < THRESHOLDS['low_recovery']).mean() * 100
    }
    
    weekly_recovery = df.groupby(df['Cycle start time'].dt.dayofweek)[METRICS['recovery']].mean()
    
    return {
        'stats': recovery_stats,
        'consistency': recovery_consistency,
        'weekly_pattern': weekly_recovery
    }

############################
# Hevy Analysis Functions #
############################

def analyze_workout_patterns(df_hevy, workout_summary):
    """
    Analyze workout patterns and progression
    """
    # Workout frequency
    plot_workout_frequency(workout_summary)
    
    # Exercise distribution
    plot_exercise_distribution(df_hevy)
    
    # Volume progression
    plot_volume_progression(workout_summary)
    
    # RPE analysis
    plot_rpe_distribution(df_hevy)

def create_strength_progression_plot(df_hevy, body_weight=85):
    """
    Create spider plot showing strength progression towards bodyweight-based goals
    """
    # Calculate target weights
    targets = {exercise: multiplier * body_weight 
              for exercise, multiplier in STRENGTH_TARGETS.items()}
    
    # Get exercise progress
    exercise_progress = get_exercise_progress(df_hevy, targets.keys())
    
    # Create visualization
    plot_strength_progression(exercise_progress, targets)
    
    # Print progress details
    print_strength_progress(exercise_progress, targets)

################################
# Combined Analysis Functions #
################################

def analyze_recovery_workout_relationship(df_physio, workout_summary):
    """
    Analyze relationship between recovery metrics and workout performance
    """
    merged_data = merge_whoop_hevy_data(df_physio, workout_summary)
    
    # Create visualizations
    plot_recovery_workout_relationships(merged_data)
    
    # Calculate correlations
    correlations = calculate_recovery_workout_correlations(merged_data)
    
    return merged_data, correlations

#############################
# Machine Learning Analysis #
#############################

def build_recovery_prediction_model(df_physio, workout_summary):
    """
    Build and evaluate recovery prediction model
    """
    # Prepare features and target
    X, y = prepare_model_data(df_physio, workout_summary)
    
    # Create and train model
    model = train_recovery_model(X, y)
    
    # Evaluate model
    evaluate_model(model, X, y)
    
    return model

def detect_anomalies(df_physio):
    """
    Detect anomalies in recovery patterns
    """
    # Prepare data for anomaly detection
    features = ['Recovery score %', 'Day Strain', 'Sleep performance %', 
                'Resting heart rate (bpm)']
    
    # Create and fit anomaly detection pipeline
    anomalies = detect_recovery_anomalies(df_physio[features])
    
    # Visualize results
    plot_anomalies(df_physio, anomalies)
    
    return anomalies

#################
# Main Function #
#################

def main():
    """
    Main execution function
    """
    # Load data
    df_physio, df_workouts, df_sleep = load_whoop_data()
    df_hevy, workout_summary = load_hevy_data()
    
    # Analyze Whoop data
    recovery_analysis = analyze_recovery_patterns(df_physio)
    
    # Analyze Hevy data
    analyze_workout_patterns(df_hevy, workout_summary)
    create_strength_progression_plot(df_hevy)
    
    # Combined analysis
    merged_data, correlations = analyze_recovery_workout_relationship(
        df_physio, workout_summary)
    
    # Machine learning analysis
    recovery_model = build_recovery_prediction_model(df_physio, workout_summary)
    anomalies = detect_anomalies(df_physio)
    
    # Save results
    save_analysis_results(recovery_analysis, correlations, anomalies)

if __name__ == "__main__":
    main()
