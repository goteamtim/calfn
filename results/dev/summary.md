# Results (results/dev)

| run | eval file | pass | 95% CI | rejections ok | false pos | missed calls | start | ms |
|---|---|---|---|---|---|---|---|---|
| base | eval_v1.jsonl | 13.3% | 10%-18% | 40/41 | 1 | 191/259 |  | 1088 |
| calfn-v1 | eval_v1.jsonl | 48.3% | 43%-54% | 38/41 | 3 | 10/259 | 65% | 1040 |
| ollama-gemma4-12b | eval_v1.jsonl | 62.7% | 57%-68% | 14/41 | 27 | 0/259 | 85% | 2928 |

## Pass rate by tag

| tag | n | base | calfn-v1 | ollama-gemma4-12b |
|---|---|---|---|---|
| date_abs | 84 | 0% | 56% | 85% |
| date_dom | 12 | 0% | 67% | 67% |
| date_monthrel | 20 | 0% | 35% | 70% |
| date_rel | 38 | 0% | 29% | 89% |
| date_today | 1 | 0% | 100% | 100% |
| date_weekday | 29 | 0% | 14% | 52% |
| distractor | 20 | 0% | 20% | 85% |
| dur | 103 | 0% | 51% | 73% |
| dur_range | 24 | 0% | 21% | 67% |
| dur_until | 22 | 0% | 41% | 59% |
| ind | 85 | 16% | 62% | 64% |
| loc | 61 | 0% | 31% | 43% |
| ood | 215 | 12% | 43% | 62% |
| rec_count | 14 | 0% | 36% | 36% |
| rec_daily | 9 | 0% | 78% | 56% |
| rec_dated | 20 | 0% | 55% | 50% |
| rec_interval | 17 | 0% | 29% | 6% |
| rec_monthly | 17 | 0% | 65% | 35% |
| rec_undated | 55 | 0% | 33% | 38% |
| rec_until | 17 | 0% | 53% | 35% |
| rec_weekly | 41 | 0% | 22% | 41% |
| rec_yearly | 8 | 0% | 25% | 38% |
| recurring | 75 | 0% | 39% | 41% |
| reject | 41 | 98% | 93% | 34% |
| reject_ambiguous | 9 | 100% | 89% | 0% |
| reject_baddate | 2 | 100% | 100% | 50% |
| reject_nodate | 7 | 100% | 100% | 0% |
| reject_notime | 7 | 86% | 86% | 14% |
| reject_offtopic | 9 | 100% | 100% | 100% |
| reject_past | 1 | 100% | 0% | 100% |
| reject_vague | 6 | 100% | 100% | 33% |
| time_12h | 166 | 0% | 43% | 72% |
| time_24h | 29 | 0% | 31% | 66% |
| time_noon | 5 | 0% | 20% | 20% |
| time_period | 18 | 0% | 67% | 72% |
| time_range | 24 | 0% | 21% | 67% |
| time_words | 17 | 0% | 47% | 29% |
| year_explicit | 3 | 0% | 33% | 100% |
