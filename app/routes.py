import os
import uuid
import math
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd

from .utils.preprocessing import load_and_prepare
from .utils.prediction import load_model, predict_dataframe
try:
    # New simple classification pipeline per spec
    from .predict import predict_with_model
except Exception:
    predict_with_model = None

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'.csv'}


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part in the request.')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected.')
        if not allowed_file(file.filename):
            return render_template('upload.html', error='Invalid file type. Please upload a .csv file.')

        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        save_name = f"{unique_id}_{filename}"
        upload_path = os.path.abspath(os.path.join(current_app.config['UPLOAD_FOLDER'], save_name))
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # Try simplified spec pipeline first (model.pkl)
        model_pkl = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'model.pkl'))
        if predict_with_model and os.path.exists(model_pkl) and os.path.getsize(model_pkl) > 0:
            try:
                df_raw = pd.read_csv(upload_path)
                simple_results = predict_with_model(df_raw)
            except Exception as e:
                return render_template('upload.html', error=f'Prediction failed: {e}')

            total = int(simple_results.get('total', 0))
            fraud_count = int(simple_results.get('fraud_count', 0))
            fraud_pct = float(simple_results.get('fraud_ratio', 0.0))
            fraud_indices = simple_results.get('fraud_indices', [])
            table_html = simple_results.get('table', '')
            probs = simple_results.get('probabilities', {})
            scaler_used = simple_results.get('scaler_used', False)
            bar_labels = simple_results.get('bar_labels', [])
            bar_values = simple_results.get('bar_values', [])

            return render_template(
                'result.html',
                total_transactions=total,
                fraud_count=fraud_count,
                fraud_percentage=fraud_pct,
                fraud_indices=fraud_indices,
                pie_labels=['Fraud', 'Non-Fraud'],
                pie_values=[fraud_count, max(0, total - fraud_count)],
                bar_labels=bar_labels,
                bar_values=bar_values,
                table_html=table_html,
                model_info={'type': 'RandomForest', 'path': 'model.pkl (trained)'},
                preproc_info={'type': 'StandardScaler' if scaler_used else 'None', 'path': 'scaler.pkl' if scaler_used else 'N/A'},
                min_prob=probs.get('min'),
                mean_prob=probs.get('mean'),
                max_prob=probs.get('max')
            )

        # Fallback to probability-based pipeline
        try:
            use_standardize = request.form.get('standardize') == 'on'
            df, meta = load_and_prepare(upload_path, standardize=use_standardize)
        except Exception as e:
            return render_template('upload.html', error=f'Failed to read/validate CSV: {e}')

        try:
            model = load_model()
        except FileNotFoundError as e:
            return render_template('upload.html', error=str(e))
        except Exception as e:
            return render_template('upload.html', error=f'Failed to load model: {e}')

        # Optional threshold from form
        th_raw = request.form.get('threshold', '').strip()
        th_val = None
        if th_raw:
            try:
                th_val = float(th_raw)
                if not (0.0 <= th_val <= 1.0):
                    raise ValueError('Threshold must be between 0 and 1')
            except Exception:
                return render_template('upload.html', error='Invalid threshold. Please enter a number between 0 and 1.')

        try:
            pred_df, summary = predict_dataframe(model, df, meta, threshold=th_val)
        except Exception as e:
            return render_template('upload.html', error=f'Prediction failed: {e}')

        # Save predictions for download
        pred_filename = f"predictions_{unique_id}.csv"
        pred_path = os.path.abspath(os.path.join(current_app.config['PREDICTIONS_FOLDER'], pred_filename))
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        pred_df.to_csv(pred_path, index=False)

        # Prepare chart data
        fraud_count = int(summary.get('fraud_count', 0))
        nonfraud_count = int(summary.get('nonfraud_count', 0))
        total = fraud_count + nonfraud_count
        fraud_pct = (fraud_count / total * 100) if total else 0.0

        # Top 10 suspicious by predicted probability then amount
        top10 = pred_df.sort_values(['fraud_probability', 'Amount'], ascending=[False, False]).head(10)
        bar_labels = [f"Txn {i+1}" for i in range(len(top10))]
        bar_values = top10['Amount'].round(2).tolist()

        # Table preview
        preview_cols = ['Time', 'Amount', 'fraud_probability']
        preview_cols = [c for c in preview_cols if c in pred_df.columns]
        preview = pred_df.head(20)[preview_cols + [c for c in pred_df.columns if c not in preview_cols][:5]].to_dict(orient='records')

        model_info = getattr(model, 'fd_model_info', {})
        preproc_info = meta.get('preprocessor', {})
        return render_template(
            'result.html',
            total_transactions=total,
            fraud_count=fraud_count,
            fraud_percentage=round(fraud_pct, 2),
            threshold=summary.get('threshold', 0.3),
            model_info=model_info,
            preproc_info=preproc_info,
            pie_labels=['Fraud', 'Non-Fraud'],
            pie_values=[fraud_count, nonfraud_count],
            bar_labels=bar_labels,
            bar_values=bar_values,
            table_rows=preview,
            download_filename=pred_filename,
            min_prob=summary.get('min_prob', 0.0),
            mean_prob=summary.get('mean_prob', 0.0),
            max_prob=summary.get('max_prob', 0.0)
        )
    # GET request: render upload page
    return render_template('upload.html')
