# Deploying Fraud Detection Web App to Render

## Step-by-Step Deployment Guide

### Prerequisites
- GitHub account with your code pushed to repository
- Render account (free tier available at https://render.com)

---

## Part 1: Prepare Your Application for Deployment

### Step 1: Create `render.yaml` Configuration File

This file is already created in your project root. It tells Render how to deploy your app.

### Step 2: Update `requirements.txt`

Make sure your requirements.txt includes gunicorn (already done):
```
Flask==3.0.3
pandas==2.2.2
scikit-learn==1.5.2
joblib==1.4.2
numpy==1.26.4
gunicorn==21.2.0
```

### Step 3: Verify Project Structure

Your project should have this structure:
```
fraud_detection_web/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── render.yaml              # Render configuration
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── predict.py
│   ├── model/
│   │   ├── model.pkl        # Trained model (IMPORTANT!)
│   │   ├── scaler.pkl       # Scaler (IMPORTANT!)
│   │   └── feature_names.txt
│   ├── templates/
│   └── static/
├── data/
├── scripts/
└── README.md
```

**CRITICAL**: Make sure `model.pkl` and `scaler.pkl` are pushed to GitHub!

---

## Part 2: Push Your Code to GitHub

### Step 4: Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Prepare for Render deployment"
git branch -M main
```

### Step 5: Push to GitHub

```bash
git remote add origin https://github.com/gowthamnachu/BIG_DATA.git
git push -u origin main
```

**Note**: You've already done this! ✓

---

## Part 3: Deploy on Render

### Step 6: Sign Up / Log In to Render

1. Go to https://render.com
2. Click **"Get Started for Free"**
3. Sign up with GitHub (recommended) or email
4. Authorize Render to access your GitHub repositories

### Step 7: Create a New Web Service

1. From Render Dashboard, click **"New +"** button (top right)
2. Select **"Web Service"**
3. Connect your GitHub repository:
   - If first time: Click **"Connect GitHub"**
   - Grant Render access to your repositories
   - Select **"gowthamnachu/BIG_DATA"** repository

### Step 8: Configure Web Service Settings

Fill in the following details:

**Basic Settings:**
- **Name**: `fraud-detection-app` (or any unique name)
- **Region**: Choose closest to you (e.g., Singapore, Oregon)
- **Branch**: `main`
- **Root Directory**: `fraud_detection_web` (IMPORTANT!)

**Build & Deploy Settings:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

**Instance Type:**
- **Free** (512 MB RAM, suitable for testing)
- Or **Starter** ($7/month, 512 MB RAM, no sleep)

**Advanced Settings (Optional):**
- **Auto-Deploy**: `Yes` (deploys automatically on git push)

### Step 9: Add Environment Variables (if needed)

Click **"Advanced"** → **"Add Environment Variable"**

Add any required variables:
- `FLASK_ENV=production`
- `PORT=10000` (Render sets this automatically)

### Step 10: Deploy!

1. Click **"Create Web Service"** button
2. Render will start building your application
3. Watch the build logs in real-time

**Build Process** (takes 3-5 minutes):
```
==> Cloning from https://github.com/gowthamnachu/BIG_DATA...
==> Downloading Python dependencies...
==> Installing requirements.txt
==> Build successful! 🎉
==> Starting service...
```

### Step 11: Access Your Deployed Application

Once deployed successfully:
1. Render will provide a URL: `https://fraud-detection-app.onrender.com`
2. Click the URL to open your live application!
3. Test by uploading a CSV file

---

## Part 4: Troubleshooting Common Issues

### Issue 1: Build Fails - Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**: Make sure `requirements.txt` is complete and in the correct location

### Issue 2: Application Crashes - Port Binding

**Error**: `Failed to bind to $PORT`

**Solution**: Update `app.py` to use Render's PORT:

```python
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### Issue 3: Model Files Not Found

**Error**: `FileNotFoundError: model.pkl not found`

**Solution**: 
1. Make sure model files are committed to Git:
   ```bash
   git add app/model/model.pkl app/model/scaler.pkl
   git commit -m "Add model files"
   git push
   ```
2. Check file paths are relative, not absolute

### Issue 4: Application Sleeps (Free Tier)

**Issue**: Free tier apps sleep after 15 minutes of inactivity

**Solutions**:
- Upgrade to Starter plan ($7/month) for 24/7 uptime
- Use a service like UptimeRobot to ping your app every 10 minutes
- Accept the sleep behavior for development/demo purposes

### Issue 5: Large Model Files

**Error**: `File too large for GitHub`

**Solution**: Use Git LFS (Large File Storage):
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add app/model/*.pkl
git commit -m "Track model files with LFS"
git push
```

---

## Part 5: Update Your Deployed App

### Step 12: Make Changes and Redeploy

1. Make code changes locally
2. Test locally: `python app.py`
3. Commit changes:
   ```bash
   git add .
   git commit -m "Update feature X"
   git push
   ```
4. Render automatically detects the push and redeploys! ✨

---

## Part 6: Monitor Your Application

### Step 13: View Logs

1. Go to Render Dashboard
2. Click on your service name
3. Click **"Logs"** tab
4. Monitor real-time application logs

### Step 14: Check Metrics

1. Click **"Metrics"** tab
2. View:
   - CPU usage
   - Memory usage
   - Request count
   - Response time

---

## Part 7: Custom Domain (Optional)

### Step 15: Add Custom Domain

1. Go to service **"Settings"**
2. Scroll to **"Custom Domains"**
3. Click **"Add Custom Domain"**
4. Enter your domain: `fraud-detection.yourdomain.com`
5. Add CNAME record in your DNS provider:
   - Name: `fraud-detection`
   - Value: `fraud-detection-app.onrender.com`
6. Wait for DNS propagation (5-60 minutes)
7. Render automatically provisions SSL certificate! 🔒

---

## Quick Reference Commands

### Local Testing
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Run locally
python app.py

# Access at http://localhost:5000
```

### Git Commands
```bash
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub (triggers auto-deploy)
git push origin main
```

### Render CLI (Optional)
```bash
# Install Render CLI
pip install render-cli

# Login
render login

# View services
render services list

# View logs
render logs -s fraud-detection-app
```

---

## Cost Breakdown

### Free Tier
- **Cost**: $0/month
- **Specs**: 512 MB RAM, 0.1 CPU
- **Limitations**: 
  - Sleeps after 15 min inactivity
  - 750 hours/month free
  - Slower cold starts
- **Best for**: Development, demos, learning

### Starter Tier (Recommended)
- **Cost**: $7/month
- **Specs**: 512 MB RAM, 0.5 CPU
- **Benefits**:
  - Always on (no sleep)
  - Faster performance
  - Priority support
- **Best for**: Production apps, portfolios

---

## Security Best Practices

1. **Never commit sensitive data**:
   - Add `.env` to `.gitignore`
   - Use Render's environment variables for secrets

2. **Use HTTPS**: Render provides free SSL automatically

3. **Set secure headers** in Flask:
   ```python
   from flask import Flask
   app = Flask(__name__)
   
   @app.after_request
   def set_secure_headers(response):
       response.headers['X-Content-Type-Options'] = 'nosniff'
       response.headers['X-Frame-Options'] = 'DENY'
       response.headers['X-XSS-Protection'] = '1; mode=block'
       return response
   ```

4. **Limit file upload size** (already implemented):
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
   ```

---

## Success Checklist

Before going live, verify:

- [ ] Model files (model.pkl, scaler.pkl) are in repository
- [ ] requirements.txt is complete and up-to-date
- [ ] app.py uses PORT from environment variable
- [ ] All file paths are relative, not absolute
- [ ] Tested file upload functionality locally
- [ ] Pushed all changes to GitHub
- [ ] Render build completed successfully
- [ ] Application accessible via Render URL
- [ ] Upload and prediction features work on live site
- [ ] Charts and visualizations render correctly

---

## Need Help?

- **Render Documentation**: https://render.com/docs
- **Render Community**: https://community.render.com
- **Flask Documentation**: https://flask.palletsprojects.com/
- **GitHub Issues**: Create issue in your repository

---

## Next Steps After Deployment

1. **Share your live URL** with team members
2. **Add URL to README.md** for easy access
3. **Test with different CSV files** to ensure reliability
4. **Monitor logs** for any errors
5. **Consider upgrading** to Starter tier for production use
6. **Add custom domain** for professional appearance
7. **Implement authentication** if needed for restricted access
8. **Add database** (PostgreSQL) for logging predictions
9. **Set up monitoring** with UptimeRobot or Pingdom
10. **Create backup strategy** for model files

---

**Congratulations! Your fraud detection app is now live! 🎉**

**Live URL**: `https://your-app-name.onrender.com`

Share this URL in your project report and presentation!
