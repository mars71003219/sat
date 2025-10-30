-- ============================================================
-- 새로운 추론 결과 스키마 (다중 모델 지원)
-- ============================================================

-- 기존 테이블 삭제
DROP TABLE IF EXISTS inference_results CASCADE;
DROP TABLE IF EXISTS inference_jobs CASCADE;
DROP TABLE IF EXISTS subsystem_inferences CASCADE;

-- ============================================================
-- 1. inference_jobs: 전체 추론 작업 관리
-- ============================================================
CREATE TABLE inference_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    satellite_id VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,  -- 'kafka_auto_trigger', 'manual', 'scheduled'
    trigger_reason VARCHAR(100),  -- 'low_battery', 'high_temp', 'thruster_active', 'periodic'
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    total_subsystems INTEGER DEFAULT 0,
    completed_subsystems INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB,
    error_message TEXT
);

CREATE INDEX idx_jobs_satellite ON inference_jobs(satellite_id);
CREATE INDEX idx_jobs_status ON inference_jobs(status);
CREATE INDEX idx_jobs_created ON inference_jobs(created_at DESC);

-- ============================================================
-- 2. subsystem_inferences: 서브시스템별 추론 결과
-- ============================================================
CREATE TABLE subsystem_inferences (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    subsystem VARCHAR(50) NOT NULL,  -- 'eps', 'thermal', 'aocs', 'comm', 'payload', 'obc'
    model_name VARCHAR(100) NOT NULL,  -- 'lstm_eps', 'transformer_thermal', 'vae_aocs'
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    
    -- 입력 데이터
    input_data JSONB NOT NULL,
    input_features TEXT[],  -- ['battery_voltage', 'battery_soc', ...]
    
    -- 추론 결과
    predictions JSONB,  -- 예측값 배열
    confidence JSONB,   -- 신뢰도 배열
    anomaly_score FLOAT,  -- 이상 감지 점수
    anomaly_detected BOOLEAN DEFAULT FALSE,
    
    -- 메트릭
    inference_time_ms FLOAT,
    model_type VARCHAR(50),  -- 'LSTM', 'Transformer', 'VAE'
    sequence_length INTEGER,
    forecast_horizon INTEGER,
    
    -- 통계
    input_mean FLOAT,
    input_std FLOAT,
    prediction_mean FLOAT,
    prediction_std FLOAT,
    
    -- 메타데이터
    triton_model_version VARCHAR(20),
    processed_by VARCHAR(50),  -- 'triton_server', 'local'
    
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX idx_subsys_job ON subsystem_inferences(job_id);
CREATE INDEX idx_subsys_subsystem ON subsystem_inferences(subsystem);
CREATE INDEX idx_subsys_model ON subsystem_inferences(model_name);
CREATE INDEX idx_subsys_anomaly ON subsystem_inferences(anomaly_detected) WHERE anomaly_detected = TRUE;
CREATE INDEX idx_subsys_created ON subsystem_inferences(created_at DESC);

-- ============================================================
-- 3. 편의를 위한 뷰: 작업 요약
-- ============================================================
CREATE OR REPLACE VIEW v_inference_job_summary AS
SELECT 
    j.job_id,
    j.satellite_id,
    j.source,
    j.trigger_reason,
    j.status as job_status,
    j.created_at,
    j.completed_at,
    j.total_subsystems,
    j.completed_subsystems,
    ROUND(EXTRACT(EPOCH FROM (COALESCE(j.completed_at, NOW()) - j.created_at))::NUMERIC, 3) as total_time_seconds,
    COUNT(si.id) as inference_count,
    COUNT(CASE WHEN si.anomaly_detected THEN 1 END) as anomalies_detected,
    AVG(si.inference_time_ms) as avg_inference_time_ms,
    json_agg(
        json_build_object(
            'subsystem', si.subsystem,
            'model', si.model_name,
            'status', si.status,
            'anomaly', si.anomaly_detected,
            'anomaly_score', si.anomaly_score
        ) ORDER BY si.created_at
    ) as subsystem_results
FROM inference_jobs j
LEFT JOIN subsystem_inferences si ON j.job_id = si.job_id
GROUP BY j.job_id, j.satellite_id, j.source, j.trigger_reason, j.status, 
         j.created_at, j.completed_at, j.total_subsystems, j.completed_subsystems;

-- ============================================================
-- 4. 편의를 위한 뷰: 이상 감지 현황
-- ============================================================
CREATE OR REPLACE VIEW v_anomaly_alerts AS
SELECT 
    si.id,
    si.job_id,
    j.satellite_id,
    si.subsystem,
    si.model_name,
    si.anomaly_score,
    si.predictions,
    si.created_at,
    j.trigger_reason,
    j.metadata
FROM subsystem_inferences si
JOIN inference_jobs j ON si.job_id = j.job_id
WHERE si.anomaly_detected = TRUE
ORDER BY si.created_at DESC;

COMMENT ON TABLE inference_jobs IS '전체 추론 작업 관리 테이블';
COMMENT ON TABLE subsystem_inferences IS '서브시스템별 개별 추론 결과';
COMMENT ON VIEW v_inference_job_summary IS '작업별 추론 결과 요약';
COMMENT ON VIEW v_anomaly_alerts IS '이상 감지 알림 뷰';
