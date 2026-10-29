# Plan

Consolidate existing CI jobs into one ten-configuration matrix. Select the existing steps per configuration, preserve step conditions, give step outputs unique IDs, and keep ccache isolated by former job ID. Preserve former timeout values. Validate workflow syntax and audit each selected configuration against the original commands. Publish a feature PR to develop and monitor GitHub outcomes.
