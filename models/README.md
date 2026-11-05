# Models

Trained files are present locally but ignored by Git. See SETUP.md for rebuilding baseline, CNNs and pretrained features. selection.json is the validation-only selection lock; docs/results/<name>.json contains preprocessing and training configuration. The selected local checkpoint is cnn_lr_low.pt. PyTorch state dictionaries load with weights_only=True. Joblib artifacts must come from trusted local training. The official pretrained download is cached under models/cache and is not committed.
