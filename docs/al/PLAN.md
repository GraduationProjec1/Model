# Active learning + chứng minh học tăng dần

Kế hoạch đầy đủ nằm ở tài liệu dự án claude.ai "Đồ án": `claude/plan-code-active-incremental-2026-11-02.md`
(lý do: `claude/ke-hoach-active-incremental-2026-11-02.md`). Nhật ký thực hiện: `NHAT-KY.md`.

Code: `autocheckout/active/` (A1-A3), `tools/al_select.py`, `tools/al_build_variant.py` (A4), `tools/al_score.py` (A5),
`update_model.py --stage al` + `scripts/run_al.sh` (A6), `tools/al_report.py` (A7), `tools/al_round2.py` (A9), `tools/forgetting_report.py` (B1),
`full_syn.NAIVE` + `tools/run_naive.py` (B2), `tools/count_params.py` (B4). Chạy trên GPU: `RUNBOOK.md`.
