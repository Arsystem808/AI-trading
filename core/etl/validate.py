# core/etl/validate.py
import great_expectations as gx

def run_gx_checks(df):
    ge_df = gx.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null("close")
    ge_df.expect_table_row_count_to_be_between(min_value=100)
    res = ge_df.validate()
    return res.success
