from mlflow.tracking import MlflowClient

client = MlflowClient()
versions = list(client.search_model_versions("name = 'aegis-causal-elasticity'"))
latest = max(versions, key=lambda v: int(v.version))
print("VERSION", latest.version)
print("RUN", latest.run_id)
if latest.run_id is None:
    raise RuntimeError("Latest registered model version has no source run")
for artifact_name in (
    "registered_model_metadata.json",
    "model_package/metadata.json",
    "diagnostics/causal_refutation_summary.json",
    "diagnostics/glm_baseline_summary.json",
):
    try:
        print("---", artifact_name)
        path = client.download_artifacts(latest.run_id, artifact_name)
        print(path)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        print(text[:4000])
    except Exception as exc:
        print("ERROR", artifact_name, type(exc).__name__, exc)
