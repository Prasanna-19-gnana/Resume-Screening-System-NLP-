# Streamlit Cloud Deployment Setup Guide

## ✅ Fixed Issues
Your app had hardcoded `localhost` URLs that only work on your local machine. These have been updated to use **Streamlit Secrets** for flexible configuration.

## 🚀 Deployment Instructions

### Step 1: Update Your `.gitignore` (If Not Already)
Make sure `.streamlit/secrets.toml` is in your `.gitignore` so local secrets are not committed to GitHub:

```
.streamlit/secrets.toml
```

### Step 2: Push Changes to GitHub
```bash
git add .
git commit -m "Fix: Configure API URLs for Streamlit Cloud deployment"
git push origin main
```

### Step 3: Connect Streamlit Cloud

1. Go to **[streamlit.io](https://streamlit.io)**
2. Click **"Sign in"** (or create account)
3. Click **"Create app"**
4. Configure:
   - **Repository**: Select your `Resume-Screening-System-NLP-` repo
   - **Branch**: `main`
   - **Main file path**: `ui/app.py`
5. Click **"Deploy"**

### Step 4: Configure Secrets in Streamlit Cloud

⚠️ **CRITICAL**: Your backend API must be deployed somewhere accessible from the internet!

**Option A: If you have a deployed API backend**

1. After deploy, go to your app settings
2. Click **"Secrets"** in the left sidebar
3. Add the API URL:
   ```toml
   api_service_url = "https://your-api-domain.com:8001"
   ```
   Replace `https://your-api-domain.com:8001` with your actual API server URL

4. Save and the app will redeploy automatically

**Option B: Quick Test (Demo Mode)**

If you don't have a backend deployed yet, you can test the UI by:
1. Running this command locally:
   ```bash
   streamlit run ui/app.py
   ```
2. Deploying just the UI to Streamlit Cloud (it will show connection errors until backend is ready)

## 📋 What Changed

### Files Modified
- **`ui/app.py`**: Updated to read `api_service_url` from Streamlit secrets

### Files Created
- **`.streamlit/secrets.toml`**: Local development configuration (not committed to git)
- **`.streamlit/config.toml`**: UI theme and display settings
- **`STREAMLIT_CLOUD_SETUP.md`**: This guide

## 🔧 Environment Variables

You can also set the API URL via environment variable (useful for Docker/production):
```bash
export API_SERVICE_URL="https://your-api-server.com:8001"
streamlit run ui/app.py
```

The app reads URLs in this priority order:
1. **Streamlit Secrets** (`api_service_url`)
2. **Environment Variable** (`API_SERVICE_URL`)
3. **Default** (`http://localhost:8001`)

## 🐛 Troubleshooting

**Error: "Connection Error" when screening resumes**
- ✅ Check that `api_service_url` is correctly set in Streamlit Cloud Secrets
- ✅ Verify your API server is running and accessible from the internet
- ✅ Check that firewall allows requests to your API

**404 Not Found**
- ✅ Ensure API endpoints match your backend:
  - `/upload-resume`
  - `/match`

**Still having issues?**
- Run locally first to isolate whether it's a frontend or backend issue
- Check Streamlit Cloud logs: View → Logs in top right corner

## 📚 Local Development

To run locally:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run ui/app.py
```

The app will automatically use `http://localhost:8001` from `.streamlit/secrets.toml`.

---

**Next Steps:**
1. ✅ Changes are already committed to your repo
2. ⏭️ Deploy to Streamlit Cloud
3. ⏭️ Set the `api_service_url` secret (once your API is live)
4. ⏭️ Test the app
