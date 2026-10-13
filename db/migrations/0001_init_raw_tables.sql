-- 0001_init_raw_tables.sql
-- 수집 단계에서 필요한 최소 raw 테이블만 생성.
-- feature_store / label_crash / news_structured / classification_result /
-- portfolio_result 등 가공·분석용 테이블은 ai 단계 진행하면서 별도 migration으로 추가.

CREATE TABLE IF NOT EXISTS stock_master (
    code            VARCHAR(10) PRIMARY KEY,   -- 종목코드 (예: 005930)
    name            VARCHAR(100) NOT NULL,     -- 종목명
    market          VARCHAR(20) NOT NULL,      -- KOSPI / KOSDAQ 등
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    listed_date     DATE,
    delisted_date   DATE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS price_daily (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    trade_date      DATE NOT NULL,
    open_price      NUMERIC(15, 2),
    high_price      NUMERIC(15, 2),
    low_price       NUMERIC(15, 2),
    close_price     NUMERIC(15, 2),
    volume          BIGINT,
    foreign_net_buy NUMERIC(18, 2),   -- 외국인 순매수
    inst_net_buy    NUMERIC(18, 2),   -- 기관 순매수
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, trade_date)
);

CREATE TABLE IF NOT EXISTS financial_statement (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    fiscal_year     INT NOT NULL,
    fiscal_quarter  INT,                -- NULL이면 연간
    report_type     VARCHAR(20),        -- 사업보고서 / 분기보고서 등
    revenue         NUMERIC(20, 2),
    operating_income NUMERIC(20, 2),
    net_income      NUMERIC(20, 2),
    total_assets    NUMERIC(20, 2),
    total_equity    NUMERIC(20, 2),
    raw_payload     JSONB,              -- OpenDART 원문 응답 보관
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (code, fiscal_year, fiscal_quarter, report_type)
);

CREATE TABLE IF NOT EXISTS disclosure (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(10) NOT NULL REFERENCES stock_master(code),
    rcept_no        VARCHAR(20) NOT NULL UNIQUE,  -- OpenDART 접수번호
    report_name     VARCHAR(200),
    submitted_at    DATE,
    url             TEXT,
    raw_payload     JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS news_raw (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(10) REFERENCES stock_master(code),  -- 종목 매칭 전이면 NULL 허용
    title           TEXT NOT NULL,
    description     TEXT,
    link            TEXT,
    published_at    TIMESTAMPTZ,
    source_query    VARCHAR(200),       -- 검색에 사용한 키워드
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (link, published_at)
);

CREATE INDEX IF NOT EXISTS idx_price_daily_code_date ON price_daily (code, trade_date);
CREATE INDEX IF NOT EXISTS idx_news_raw_code ON news_raw (code);
CREATE INDEX IF NOT EXISTS idx_disclosure_code ON disclosure (code);
