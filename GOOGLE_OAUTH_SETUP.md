# Google OAuth Setup Guide

## Step 1: Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **OAuth 2.0 Client ID**
5. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: LearnFlow
   - User support email: your email
   - Developer contact: your email
6. For Application type, select **Web application**
7. Add these **Authorized redirect URIs**:
   ```
   http://localhost:8000/accounts/google/login/callback/
   http://127.0.0.1:8000/accounts/google/login/callback/
   ```
8. Click **Create**
9. **Copy the Client ID and Client Secret** (you'll need these)

## Step 2: Configure in Django Admin

1. Make sure your Django server is running:
   ```bash
   python manage.py runserver
   ```

2. Create a superuser if you haven't already:
   ```bash
   python manage.py createsuperuser
   ```

3. Go to `http://localhost:8000/admin` and login

4. Navigate to **Sites** section:
   - Click on the existing site (example.com)
   - Change **Domain name** to: `localhost:8000`
   - Change **Display name** to: `LearnFlow`
   - Click **Save**

5. Navigate to **Social applications** → **Add social application**:
   - **Provider**: Select `Google`
   - **Name**: `Google OAuth`
   - **Client id**: Paste your Client ID from Google Console
   - **Secret key**: Paste your Client Secret from Google Console
   - **Sites**: Move `localhost:8000` from Available to Chosen
   - Click **Save**

## Step 3: Update Frontend Login Button

The button is already configured to redirect to:
```
http://localhost:8000/api/auth/google/login/
```

However, the correct allauth URL is:
```
http://localhost:8000/accounts/google/login/
```

I'll fix this in the Login.js file.

## Step 4: Test the Flow

1. Navigate to `http://localhost:3000/login`
2. Click **Continue with Google**
3. You'll be redirected to Google's login page
4. After authentication, you'll be redirected back to your app
5. The user will be automatically created and logged in

## Troubleshooting

**Error: redirect_uri_mismatch**
- Make sure the redirect URI in Google Console exactly matches: `http://localhost:8000/accounts/google/login/callback/`

**Error: Site matching query does not exist**
- Make sure you've configured the Site in Django admin with domain `localhost:8000`

**Error: Social application not found**
- Make sure you've added the Google social application in Django admin
- Verify the Client ID and Secret are correct
