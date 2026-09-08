# Security

Use trusted local checkpoints, safetensors for model weights and `weights_only=True` for compatible range tables. Legacy training checkpoints contain optimizer/RNG pickle state: do not load untrusted snapshots. Do not expose a Ray Dashboard publicly or include cluster endpoints, tokens or proxy credentials in releases. The release wrapper never publishes assets automatically.

Before public release, add a maintainer security contact or enable private GitHub vulnerability reporting in the chosen repository. Report credential exposures privately; rotate the credential rather than relying on removal from the latest commit.
