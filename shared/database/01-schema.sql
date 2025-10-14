-- PostgreSQL Schema for AI Inference System
-- This file is automatically executed by PostgreSQL on first startup

-- Create inference_results table
CREATE TABLE IF NOT EXISTS inference_results (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    predictions JSONB,
    confidence JSONB,
    metrics JSONB,
    metadata JSONB,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_inference_job_id ON inference_results(job_id);
CREATE INDEX idx_inference_model_name ON inference_results(model_name);
CREATE INDEX idx_inference_status ON inference_results(status);
CREATE INDEX idx_inference_created_at ON inference_results(created_at);

-- Create sensor_data table
CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    location VARCHAR(255),
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    vibration FLOAT,
    flow_rate FLOAT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sensor_id ON sensor_data(sensor_id);
CREATE INDEX idx_sensor_type ON sensor_data(sensor_type);
CREATE INDEX idx_sensor_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_sensor_location ON sensor_data(location);

-- Create timeseries_datasets table
CREATE TABLE IF NOT EXISTS timeseries_datasets (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(100) NOT NULL,
    description TEXT,
    data_points JSONB NOT NULL,
    labels JSONB,
    features JSONB,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dataset_name ON timeseries_datasets(dataset_name);
CREATE INDEX idx_dataset_type ON timeseries_datasets(dataset_type);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_name VARCHAR(255),
    metadata JSONB,
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_metrics_model_name ON model_metrics(model_name);
CREATE INDEX idx_metrics_type ON model_metrics(metric_type);
CREATE INDEX idx_metrics_recorded_at ON model_metrics(recorded_at);

-- Create activity_logs table
CREATE TABLE IF NOT EXISTS activity_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_logs_user_id ON activity_logs(user_id);
CREATE INDEX idx_logs_action_type ON activity_logs(action_type);
CREATE INDEX idx_logs_timestamp ON activity_logs(timestamp);

-- Create views for analytics

-- Prediction history view
CREATE OR REPLACE VIEW prediction_history AS
SELECT
    ir.job_id,
    ir.model_name,
    ir.status,
    ir.predictions,
    ir.confidence,
    ir.created_at,
    ir.completed_at,
    EXTRACT(EPOCH FROM (ir.completed_at - ir.created_at)) as processing_time_seconds
FROM inference_results ir
WHERE ir.status = 'completed'
ORDER BY ir.created_at DESC;

-- Model performance summary view
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    model_name,
    COUNT(*) as total_inferences,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_inferences,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_inferences,
    ROUND(
        100.0 * COUNT(CASE WHEN status = 'completed' THEN 1 END) / COUNT(*),
        2
    ) as success_rate,
    AVG(
        CASE
            WHEN completed_at IS NOT NULL
            THEN EXTRACT(EPOCH FROM (completed_at - created_at))
        END
    ) as avg_processing_time_seconds,
    MIN(created_at) as first_inference,
    MAX(created_at) as last_inference
FROM inference_results
GROUP BY model_name
ORDER BY total_inferences DESC;
