# Dataset sync via GCS

The canonical dataset copy lives in the `padel-cv-dataset` Google Cloud Storage bucket. Two helper scripts keep the repo in sync with that bucket while preserving the folder structure under the `video/` prefix.

## Upload

1. Authenticate locally once: `gcloud auth application-default login`.
2. From the project root, run `scripts/upload-dataset-gcs --dataset ./dataset --skip-existing` to find every `.mp4` under `dataset/` and push it to `gs://padel-cv-dataset/video/<relative-path>`.
3. Use `--dry-run` for a quick preview or drop `--skip-existing` if you need to re-upload everything.

## Download

1. Inside Colab (or any other machine), run `gcloud auth login` so the runtime can access the bucket.
2. Pull the videos with `scripts/download-dataset-gcs --destination dataset --skip-existing` to mirror the `video/` prefix locally.
3. Pass `--dry-run` to list pending transfers without downloading.

These scripts only touch `.mp4` assets under `video/`, leaving room for other artifacts (like skeleton JSON) to live under separate prefixes.
