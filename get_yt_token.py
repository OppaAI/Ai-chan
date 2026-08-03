from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    '/home/oppa-ai/.aiko/client_secret.json',
    scopes=['https://www.googleapis.com/auth/youtube.upload']
)

flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'

auth_url, state = flow.authorization_url(
    prompt='consent',
    access_type='offline'
)

print(f"\n🔗 Open this URL in your browser:\n{auth_url}\n")

code = input("Paste the code here: ").strip()
flow.fetch_token(code=code)

# Use flow.credentials instead
creds = flow.credentials

print(f"\n✅ New refresh token: {creds.refresh_token}")
with open('youtube_token.json', 'w') as f:
    import json
    json.dump({
        'refresh_token': creds.refresh_token,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'token_uri': creds.token_uri
    }, f)
    print(f"✅ Saved to youtube_token.json")
