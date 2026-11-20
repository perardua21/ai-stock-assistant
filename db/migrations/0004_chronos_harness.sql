-- 0004_chronos_harness.sql
-- 구조 변경 (Chronos-2 + 급락 분류기 + Self-evolving 하네스) 에 맞춘 테이블 정리.
-- 여러 번 실행해도 결과가 같도록 IF EXISTS / IF NOT EXISTS 로 작성.

-- 1) 가격 테이블 보강: 거래대금, 시가총액 (pykrx 수집)
ALTER TABLE price_daily ADD COLUMN IF NOT EXISTS trading_value NUMERIC(20, 2);
ALTER TABLE price_daily ADD COLUMN IF NOT EXISTS market_cap    NUMERIC(22, 2);

-- 2) 지수 일봉 (KOSPI = 1001). Chronos-2 공변량과 시장 피처에 사용
CREATE TABLE IF NOT EXISTS index_daily (
    index_code      VARCHAR(10) NOT NULL,
    trade_date      DATE NOT NULL,
    open_price      NUMERIC(12, 2),
    high_price      NUMERIC(12, 2),
    low_price       NUMERIC(12, 2),
    close_price     NUMERIC(12, 2),
    volume          BIGINT,
    trading_value   NUMERIC(22, 2),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (index_code, trade_date)
);

-- 3) Chronos-2 예측 분포를 요약한 가격 피처 (종목·기준일·모델별 1행)
CREATE TABLE IF NOT EXISTS forecast_feature (
    code            VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    base_date       DATE NOT NULL,
    model_name      VARCHAR(60) NOT NULL,      -- 예: chronos-2, chronos-2+cov
    horizon         SMALLINT NOT NULL,
    last_close      NUMERIC(15, 2),
    p_crash         NUMERIC(6, 5),             -- 예측 분포로 계산한 급락(-10%) 확률
    q05_ret_min     NUMERIC(8, 5),             -- horizon 동안 하위 5% 분위수 수익률의 최솟값
    q10_ret         NUMERIC(8, 5),             -- horizon 마지막 날 하위 10% 분위수 수익률
    q50_ret         NUMERIC(8, 5),
    q90_ret         NUMERIC(8, 5),
    spread          NUMERIC(8, 5),             -- q90 - q10 (불확실성)
    skew            NUMERIC(8, 5),             -- 위/아래 꼬리 비대칭
    raw_quantiles   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (code, base_date, model_name)
);
CREATE INDEX IF NOT EXISTS idx_forecast_feature_date ON forecast_feature (base_date);

-- 4) 기존 멀티모달·뉴스 XAI 테이블 제거 (뉴스는 학습에 쓰지 않음)
DROP TABLE IF EXISTS news_token_importance;
DROP TABLE IF EXISTS news_ablation;
DROP TABLE IF EXISTS explanation_log;

-- classification_result / feature_importance 는 그대로 쓴다
-- (LightGBM 급락 확률 + SHAP 기여도)

-- 5) 하네스: 에이전트 판단 기록 (메모리 역할도 함)
CREATE TABLE IF NOT EXISTS agent_decision (
    id                  BIGSERIAL PRIMARY KEY,
    code                VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    run_date            DATE NOT NULL,
    model_probability   NUMERIC(6, 5),          -- 분류기가 준 급락 확률
    agent_probability   NUMERIC(6, 5),          -- 에이전트가 뉴스·공시를 보고 조정한 확률
    judgment            VARCHAR(10) NOT NULL CHECK (judgment IN ('high', 'medium', 'low')),
    reasons             JSONB NOT NULL,         -- 판단 근거 문장 목록
    evidence            JSONB,                  -- 툴 호출 기록 (뉴스·공시·지표)
    prompt_version      VARCHAR(40),
    skill_version       VARCHAR(40),
    llm_model           VARCHAR(60),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, run_date, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_agent_decision_date ON agent_decision (run_date);

-- 6) 5거래일 뒤 채점 결과
CREATE TABLE IF NOT EXISTS agent_score (
    decision_id     BIGINT PRIMARY KEY REFERENCES agent_decision(id) ON DELETE CASCADE,
    scored_date     DATE NOT NULL,
    min_return      NUMERIC(8, 5),
    is_crash        BOOLEAN NOT NULL,
    correct         BOOLEAN NOT NULL,
    brier           NUMERIC(6, 5),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 7) 스킬·프롬프트 개선 이력 (Self-evolving)
CREATE TABLE IF NOT EXISTS skill_revision (
    id              BIGSERIAL PRIMARY KEY,
    skill_name      VARCHAR(60) NOT NULL,
    base_version    VARCHAR(40),
    new_version     VARCHAR(40),
    status          VARCHAR(10) NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'applied', 'rejected')),
    proposal_path   TEXT,
    rationale       TEXT,
    metrics         JSONB,                       -- 개선안을 만들 때 본 채점 통계
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 8) 포트폴리오 추천 결과
CREATE TABLE IF NOT EXISTS portfolio_result (
    id                  BIGSERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    code                VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    weight              NUMERIC(6, 5) NOT NULL,
    expected_return     NUMERIC(8, 5),
    crash_probability   NUMERIC(6, 5),
    params              JSONB,                   -- λ, γ, 상한 등
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_date, code)
);
