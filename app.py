from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Get port from environment variable (Render sets this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in production (on Render)
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=not is_production)
