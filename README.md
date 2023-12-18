# -HOW-TO-COMBINE-SIMILARITY-METRICS-FOR-RE-IDENTIFICATION-PROBLEM-

## Test

```bash
bash scripts/test.sh
```
You can modify `scripts/test.sh` to change metric options at `TEST.METRICS "('')"`. Options are as follows:
- `fusion_emd` for using fusion (cosine + EMD) as a metric.
- `fusion_centroid` for using fusion (cosine + centroid) as a metric.
- `centroid` for using only centroid as a metric.

We using `cosine` as default metric.
