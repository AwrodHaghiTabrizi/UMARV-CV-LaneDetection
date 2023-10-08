# DropBox

Dropbox access tokens are needed to download and upload files from our scripts and notebooks.

## DropBox Initialization

1. Go to dropbox.umich.edu
2. Set up your account
3. Get access to the DropBox (ask team lead)
4. Go to https://www.dropbox.com/developers/apps
5. Click "Create App"
6. Click "Scoped Access"
7. Click "Full DropBox"
8. Name the app "UMARV CV {your_umich_id}"
9. Click "I Agree"
10. Click "Create App"
11. Click on the "Permissions" tab
12. Place checkmarks on the following sections:
    1. account_info.read
    2. files.metadata.write
    3. files.metadata.read
    4. files.content.write
    5. files.content.read

## DropBox Generate Access Token

1. Go to https://www.dropbox.com/developers/apps
2. Click app "UMARV CV {your_um_id}"
3. Click "Generate access toekn"

# GitHub

GitHub access tokens are needed to push changes in the Google Colab and LambdaLabs environments.

## GitHub Generate Access Token

1. Go to "https://github.com/settings/tokens"
2. Click "Tokens (classic)"
3. Click "Generate new token"
4. Click "Generate new token (classic)"
5. Note : "UMARV CV"
6. Check on "repo"
7. Click "Generate token"
