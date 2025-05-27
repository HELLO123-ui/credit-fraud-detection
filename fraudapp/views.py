import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns

# ✅ Use a non-GUI backend for macOS to avoid thread crash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import HttpResponse



# Load model
model = joblib.load('fraudapp/best_tree_model.pkl')
REQUIRED_COLUMNS = [f'A{i}' for i in range(1, 15)]

# Global cache for uploaded dataframe
uploaded_df = None


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'fraudapp/login.html', {'error': 'Invalid credentials'})
    return render(request, 'fraudapp/login.html')


def logout_view(request):
    logout(request)
    return redirect('home')


def home(request):
    name = request.user.username if request.user.is_authenticated else None
    return render(request, 'fraudapp/home.html', {'username': name})


def about(request):
    return render(request, 'fraudapp/about.html')


@login_required(login_url='/login/')
def predict(request):
    if request.method == 'POST':
        try:
            input_data = [float(request.POST.get(f'A{i}', 0)) for i in range(1, 15)]
            prediction = model.predict([input_data])[0]
            result = "⚠️ Fraud Detected" if prediction == 1 else "✅ No Fraud Detected"
            return render(request, 'fraudapp/result.html', {'result': result})
        except (ValueError, TypeError):
            return render(request, 'fraudapp/predict.html', {
                'range': range(1, 15),
                'error': "❌ Please fill in all the fields with valid numbers."
            })
    return render(request, 'fraudapp/predict.html', {'range': range(1, 15)})


@login_required(login_url='/login/')
def dashboard(request):
    global uploaded_df
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        file_name = request.POST.get('file_name', '').strip()
        uploaded_file = request.FILES.get('file')

        if not file_name:
            error_message = "❌ Please enter a file name."
        elif not uploaded_file:
            error_message = "❌ Please choose a CSV file to upload."
        else:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = [col.upper() for col in df.columns]
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols:
                    error_message = "❌ Missing required columns in CSV: " + ", ".join(missing_cols)
                else:
                    df['PREDICTION'] = model.predict(df[REQUIRED_COLUMNS])
                    df['PREDICTION'] = df['PREDICTION'].map({1: "⚠️ Fraud", 0: "✅ No Fraud"})
                    prediction_result = df.to_html(classes='table table-bordered', index=False)
                    uploaded_df = df.copy()
            except Exception as e:
                error_message = f"❌ Error processing file: {str(e)}"

    return render(request, 'fraudapp/dashboard.html', {
        'username': request.user.username,
        'result_table': prediction_result,
        'error': error_message
    })


@login_required(login_url='/login/')
def analysis(request):
    global uploaded_df
    if uploaded_df is None:
        return HttpResponse("No CSV file uploaded for analysis.")

    df = uploaded_df.copy()
    numeric_df = df[REQUIRED_COLUMNS]

    # Statistics
    stats_table = numeric_df.describe().T.round(2).to_html(classes="table table-bordered", float_format="%.2f")

    # Fraud percent if available
    fraud_percent = None
    if 'PREDICTION' in df.columns:
        counts = df['PREDICTION'].value_counts(normalize=True) * 100
        fraud_percent = counts.to_dict()

    # Heatmap
    heatmap_path = os.path.join(settings.MEDIA_ROOT, 'heatmap.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return render(request, 'fraudapp/analysis.html', {
        'username': request.user.username,
        'statistics': stats_table,
        'fraud_percent': fraud_percent,
        'heatmap_url': settings.MEDIA_URL + 'heatmap.png',
        'data_table': df.head(50).to_html(classes="table table-striped", index=False)
    })

