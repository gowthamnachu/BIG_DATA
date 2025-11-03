from flask import Flask
from flask import render_template
from .routes import bp as main_bp
import os


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=False)

    # Basic config
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
    app.config['PREDICTIONS_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB max upload

    # Ensure folders exist
    os.makedirs(os.path.abspath(app.config['UPLOAD_FOLDER']), exist_ok=True)
    os.makedirs(os.path.abspath(app.config['PREDICTIONS_FOLDER']), exist_ok=True)

    # Register blueprints
    app.register_blueprint(main_bp)

    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        return render_template('upload.html', error='File too large. Max 256MB.'), 413

    @app.errorhandler(404)
    def not_found(e):
        return render_template('index.html'), 404

    return app
