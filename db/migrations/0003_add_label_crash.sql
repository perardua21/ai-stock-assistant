-- 급락 정답 레이블 (ai/labeling/build_crash_label.py 가 생성해 적재)
CREATE TABLE IF NOT EXISTS label_crash (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    base_date DATE NOT NULL,
    horizon_days SMALLINT NOT NULL DEFAULT 5,
    threshold NUMERIC(5,4) NOT NULL DEFAULT -0.10,
    is_crash BOOLEAN NOT NULL,
    min_return NUMERIC(6,5) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, base_date, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_label_crash_code_date ON label_crash (code, base_date);
