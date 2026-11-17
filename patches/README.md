# 本地上游补丁

## shopping-grpo-runtime-manifest.patch

适用于上游 commit `dc6c9af8e48c8e6325101b39d1d20060f69e4221`。

该 commit 中 `data/environment.json` 声明的 `web_agent_text_env.py` SHA-256 为 `448f…`, 但同一 commit 的实际 Git blob SHA-256 为 `d6db…`。上游 `scripts/check_grpo_runtime.py` 会因此在加载模型前拒绝环境。

补丁只把清单中的该文件哈希改成经 `git show HEAD:<path> | sha256sum` 验证的实际值；不改环境逻辑、奖励、任务或模型行为。应用后必须重新运行上游运行时预检并保存输出。
