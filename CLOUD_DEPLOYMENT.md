# Android and cloud deployment

The checker is a responsive Streamlit web app. On Android, it runs in Chrome and can be added to the home screen for app-like access. The app prioritizes selecting an existing JPG/PNG from Gallery or Files.

## Deploy with Streamlit Community Cloud

1. Merge the tested GitHub pull request into `main`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Connect the GitHub account that owns `yslim1520/image-capture-and-measure`.
4. If the repository remains private, grant Streamlit access to private repositories.
5. Choose **Create app**, then select:
   - Repository: `yslim1520/image-capture-and-measure`
   - Branch: `main`
   - Entrypoint: `app.py`
6. In Advanced settings, select Python 3.12.
7. Deploy, then choose the app's sharing setting.

The repository already includes `requirements.txt` and `.streamlit/config.toml`, which Community Cloud reads during deployment.

Official guidance:

- [Deploy a Streamlit app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Connect GitHub and authorize private repositories](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)
- [Configure app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Configure app sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

## Use it on Android

1. Open the deployed `streamlit.app` link in Chrome.
2. Open Chrome's menu and choose **Add to Home screen**.
3. Launch the checker from the new home-screen icon.
4. Tap **Select photo from phone or computer**, then choose the original image from Gallery or Files.
5. Follow the four in-app stages: Detect, Correct rings, Calibrate & review, and Export.

## Photo guidance

- Use the original high-resolution landscape image, not a screenshot or a compressed messaging-app copy.
- Use even daylight, avoid strong glare and deep shadows, clean the lens, and hold the phone steady.
- Keep the camera as square as practical to the log ends and include the complete stack.
- Before taking the photo, choose and visibly mark one or two reference logs.
- Measure each reference across the outside bark and record the actual diameter in inches.
- Position two reference logs apart from each other when practical so scale disagreement is easier to detect.

## Privacy note

On Community Cloud, selected photos are uploaded to the cloud app for processing. This application does not intentionally save uploaded photos to persistent storage, but access and retention are still subject to the cloud provider's service and sharing settings. Keep the app private if the photos are sensitive.

