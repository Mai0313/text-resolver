# Datasets should be replaced here

## You need to split your dataset into train and val, and put it into the corresponding folder

### The dataset should be organized as follows:

#### Before you setup, it should looks like...

```
data
├── train
│   ├── train.zip
├── val
│   ├── val.zip
└── test
    ├── test.zip
```

#### After you parse the dataset, it should looks like...

```
data
├── train
│   ├── train.zip
│   └── train.npz
├── val
│   ├── val.zip
│   └── val.npz
└── test
    ├── test.zip
    └── test.npz
```

# Introduction of DVC

## Step 1: Install DVC

```bash
pip install dvc dvc-gdrive
```

## Step 2: Create a Google Drive API Key

1. Go to https://console.developers.google.com/apis/credentials
2. Click on "Create Credentials" and select "OAuth client ID"
3. Select "Desktop app" as the application type
4. Click on "Create"
5. Click on "Download JSON" and save the file as `client_secrets.json` in the same directory as this README

## Step 3: Authenticate DVC with Google Drive

```bash
dvc remote add --default gdrive gdrive://1nR6ZZU7gpNDyrKhz8hOC_vATWGP5hmMG
dvc remote modify gdrive gdrive_acknowledge_abuse true
```

## Step 4: Add and Push Data to Google Drive

```bash
dvc add data
dvc push
```
