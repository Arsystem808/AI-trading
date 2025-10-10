# core/monitoring/drift_evidently.py
from evidently.report import Report
from evidently.metrics import DataDriftPreset

def drift_report(ref_df, cur_df, out_html="reports/drift.html"):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(out_html)
    return report.as_dict()
