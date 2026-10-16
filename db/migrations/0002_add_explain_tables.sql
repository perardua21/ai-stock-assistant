-- 0002_add_explain_tables.sql
-- Classification 결과 + XAI(SHAP · Integrated Gradients · Ablation) 저장용 테이블

CREATE TABLE IF NOT EXISTS classification_result (
    id                  BIGSERIAL PRIMARY KEY,
    code                VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    run_date            DATE NOT NULL,
    crash_probability   NUMERIC(6, 5) NOT NULL,
    model_version       VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, run_date, model_version)
);

-- 정형 데이터 영향 -- SHAP
CREATE TABLE IF NOT EXISTS feature_importance (
    id                  BIGSERIAL PRIMARY KEY,
    classification_id   BIGINT NOT NULL REFERENCES classification_result(id),
    feature_name        VARCHAR(100) NOT NULL,
    shap_value          NUMERIC(12, 6),
    contribution_pct    NUMERIC(6, 2),        -- 전체 기여도 중 비율(%)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 뉴스 중요 표현 -- Integrated Gradients
CREATE TABLE IF NOT EXISTS news_token_importance (
    id                  BIGSERIAL PRIMARY KEY,
    classification_id   BIGINT NOT NULL REFERENCES classification_result(id),
    news_id             BIGINT NOT NULL REFERENCES news_raw(id),
    token                VARCHAR(100) NOT NULL,
    ig_score            NUMERIC(12, 6),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 뉴스 1건 제거 시 변화 -- Ablation
CREATE TABLE IF NOT EXISTS news_ablation (
    id                  BIGSERIAL PRIMARY KEY,
    classification_id   BIGINT NOT NULL REFERENCES classification_result(id),
    news_id             BIGINT NOT NULL REFERENCES news_raw(id),
    prob_with_all       NUMERIC(6, 5),
    prob_without        NUMERIC(6, 5),
    delta               NUMERIC(6, 5),         -- prob_with_all - prob_without
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GPT가 생성한 최종 추천 사유 (XAI 근거 통합 결과)
CREATE TABLE IF NOT EXISTS explanation_log (
    id                  BIGSERIAL PRIMARY KEY,
    classification_id   BIGINT NOT NULL REFERENCES classification_result(id),
    explanation_text    TEXT NOT NULL,
    model               VARCHAR(50),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feature_importance_cls ON feature_importance (classification_id);
CREATE INDEX IF NOT EXISTS idx_news_token_importance_cls ON news_token_importance (classification_id);
CREATE INDEX IF NOT EXISTS idx_news_ablation_cls ON news_ablation (classification_id);
CREATE INDEX IF NOT EXISTS idx_explanation_log_cls ON explanation_log (classification_id);
