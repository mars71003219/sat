-- Seed data for AI Inference System
-- This file is automatically executed by PostgreSQL after schema creation

-- Insert sample sensor data
INSERT INTO sensor_data (sensor_id, sensor_type, location, timestamp, temperature, humidity, pressure, vibration, flow_rate, metadata)
VALUES
    ('SENSOR_001', 'temperature', 'Building A - Floor 1', NOW() - INTERVAL '1 day', 22.5, 45.2, 1013.25, NULL, NULL, '{"zone": "production", "calibration_date": "2024-01-01"}'),
    ('SENSOR_002', 'humidity', 'Building A - Floor 1', NOW() - INTERVAL '1 day', NULL, 48.7, NULL, NULL, NULL, '{"zone": "production"}'),
    ('SENSOR_003', 'pressure', 'Building A - Floor 2', NOW() - INTERVAL '1 day', NULL, NULL, 1012.80, NULL, NULL, '{"zone": "storage"}'),
    ('SENSOR_004', 'vibration', 'Building B - Floor 1', NOW() - INTERVAL '1 day', NULL, NULL, NULL, 0.05, NULL, '{"zone": "machinery", "critical": true}'),
    ('SENSOR_005', 'flow_rate', 'Building B - Floor 1', NOW() - INTERVAL '1 day', NULL, NULL, NULL, NULL, 125.3, '{"zone": "machinery", "unit": "L/min"}'),
    ('SENSOR_001', 'temperature', 'Building A - Floor 1', NOW() - INTERVAL '23 hours', 23.1, 44.8, 1013.50, NULL, NULL, '{"zone": "production"}'),
    ('SENSOR_002', 'humidity', 'Building A - Floor 1', NOW() - INTERVAL '23 hours', NULL, 47.5, NULL, NULL, NULL, '{"zone": "production"}'),
    ('SENSOR_003', 'pressure', 'Building A - Floor 2', NOW() - INTERVAL '23 hours', NULL, NULL, 1013.10, NULL, NULL, '{"zone": "storage"}'),
    ('SENSOR_004', 'vibration', 'Building B - Floor 1', NOW() - INTERVAL '23 hours', NULL, NULL, NULL, 0.04, NULL, '{"zone": "machinery", "critical": true}'),
    ('SENSOR_005', 'flow_rate', 'Building B - Floor 1', NOW() - INTERVAL '23 hours', NULL, NULL, NULL, NULL, 123.7, '{"zone": "machinery", "unit": "L/min"}'),
    ('SENSOR_001', 'temperature', 'Building A - Floor 1', NOW() - INTERVAL '22 hours', 22.8, 45.5, 1013.75, NULL, NULL, '{"zone": "production"}'),
    ('SENSOR_002', 'humidity', 'Building A - Floor 1', NOW() - INTERVAL '22 hours', NULL, 46.2, NULL, NULL, NULL, '{"zone": "production"}'),
    ('SENSOR_006', 'temperature', 'Building C - Lab', NOW() - INTERVAL '1 day', 20.5, 50.0, 1013.00, NULL, NULL, '{"zone": "laboratory", "controlled_environment": true}'),
    ('SENSOR_007', 'vibration', 'Building B - Floor 2', NOW() - INTERVAL '1 day', NULL, NULL, NULL, 0.08, NULL, '{"zone": "heavy_machinery", "critical": true}'),
    ('SENSOR_008', 'flow_rate', 'Building A - Basement', NOW() - INTERVAL '1 day', NULL, NULL, NULL, NULL, 98.5, '{"zone": "cooling_system", "unit": "L/min"}');

-- Insert sample timeseries datasets
INSERT INTO timeseries_datasets (dataset_name, dataset_type, description, data_points, labels, features, metadata)
VALUES
    (
        'temperature_pattern_linear',
        'linear_trend',
        'Linear increasing temperature pattern',
        '[10.5, 12.3, 14.1, 15.9, 17.7, 19.5, 21.3, 23.1, 24.9, 26.7]',
        '[15.5, 17.3, 19.1, 20.9, 22.7]',
        '{"trend": "increasing", "slope": 1.8, "intercept": 10.5}',
        '{"source": "simulated", "quality": "high"}'
    ),
    (
        'humidity_pattern_seasonal',
        'seasonal',
        'Seasonal humidity variation pattern',
        '[45.0, 50.5, 58.2, 62.8, 59.4, 52.1, 46.3, 44.8, 48.5, 55.7]',
        '[60.2, 57.8, 51.5, 47.2, 49.8]',
        '{"period": 4, "amplitude": 18.0, "baseline": 52.0}',
        '{"source": "simulated", "quality": "high"}'
    ),
    (
        'pressure_pattern_stable',
        'random_walk',
        'Atmospheric pressure with random fluctuations',
        '[1013.25, 1013.50, 1013.10, 1012.95, 1013.30, 1013.65, 1013.40, 1013.20, 1013.55, 1013.35]',
        '[1013.15, 1013.45, 1013.60, 1013.25, 1013.50]',
        '{"volatility": 0.3, "mean": 1013.35}',
        '{"source": "simulated", "quality": "medium"}'
    ),
    (
        'vibration_pattern_cyclical',
        'cyclical',
        'Machine vibration cyclical pattern',
        '[0.02, 0.05, 0.08, 0.10, 0.08, 0.05, 0.02, 0.04, 0.07, 0.09]',
        '[0.08, 0.06, 0.03, 0.05, 0.08]',
        '{"frequency": 0.5, "amplitude": 0.08, "baseline": 0.05}',
        '{"source": "machine_monitor", "critical_threshold": 0.15}'
    ),
    (
        'flow_rate_exponential',
        'exponential',
        'Exponential increasing flow rate',
        '[100.0, 105.1, 110.5, 116.2, 122.1, 128.4, 135.0, 141.9, 149.2, 156.8]',
        '[164.9, 173.3, 182.2, 191.5, 201.3]',
        '{"growth_rate": 0.05, "initial_value": 100.0}',
        '{"source": "flow_meter", "unit": "L/min"}'
    );

-- Insert sample model metrics
INSERT INTO model_metrics (model_name, metric_type, metric_value, dataset_name, metadata, recorded_at)
VALUES
    ('lstm_timeseries', 'accuracy', 0.945, 'temperature_pattern_linear', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '7 days'),
    ('lstm_timeseries', 'mse', 0.023, 'temperature_pattern_linear', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '7 days'),
    ('lstm_timeseries', 'mae', 0.112, 'temperature_pattern_linear', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '7 days'),
    ('lstm_timeseries', 'r2_score', 0.932, 'temperature_pattern_linear', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '7 days'),
    ('lstm_timeseries', 'inference_time', 0.045, 'temperature_pattern_linear', '{"device": "gpu", "batch_size": 8}', NOW() - INTERVAL '7 days'),
    ('moving_average', 'accuracy', 0.878, 'humidity_pattern_seasonal', '{"window_size": 5}', NOW() - INTERVAL '6 days'),
    ('moving_average', 'mse', 0.056, 'humidity_pattern_seasonal', '{"window_size": 5}', NOW() - INTERVAL '6 days'),
    ('moving_average', 'mae', 0.185, 'humidity_pattern_seasonal', '{"window_size": 5}', NOW() - INTERVAL '6 days'),
    ('moving_average', 'r2_score', 0.845, 'humidity_pattern_seasonal', '{"window_size": 5}', NOW() - INTERVAL '6 days'),
    ('moving_average', 'inference_time', 0.002, 'humidity_pattern_seasonal', '{"device": "cpu"}', NOW() - INTERVAL '6 days'),
    ('lstm_timeseries', 'accuracy', 0.952, 'humidity_pattern_seasonal', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '5 days'),
    ('lstm_timeseries', 'mse', 0.019, 'humidity_pattern_seasonal', '{"epochs": 100, "batch_size": 32}', NOW() - INTERVAL '5 days'),
    ('moving_average', 'accuracy', 0.891, 'temperature_pattern_linear', '{"window_size": 3}', NOW() - INTERVAL '4 days'),
    ('moving_average', 'mse', 0.042, 'temperature_pattern_linear', '{"window_size": 3}', NOW() - INTERVAL '4 days'),
    ('lstm_timeseries', 'accuracy', 0.938, 'flow_rate_exponential', '{"epochs": 150, "batch_size": 16}', NOW() - INTERVAL '3 days');

-- Insert sample activity logs
INSERT INTO activity_logs (user_id, action_type, resource_type, resource_id, details, ip_address, user_agent, timestamp)
VALUES
    ('user_001', 'inference_submit', 'inference_job', 'job_12345', '{"model_name": "lstm_timeseries", "data_size": 10}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', NOW() - INTERVAL '2 days'),
    ('user_001', 'inference_complete', 'inference_job', 'job_12345', '{"status": "completed", "duration": 0.45}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', NOW() - INTERVAL '2 days' + INTERVAL '1 second'),
    ('user_002', 'inference_submit', 'inference_job', 'job_12346', '{"model_name": "moving_average", "data_size": 10}', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', NOW() - INTERVAL '2 days' + INTERVAL '5 minutes'),
    ('user_002', 'inference_complete', 'inference_job', 'job_12346', '{"status": "completed", "duration": 0.02}', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', NOW() - INTERVAL '2 days' + INTERVAL '5 minutes' + INTERVAL '200 milliseconds'),
    ('user_003', 'model_query', 'model_info', 'lstm_timeseries', '{"action": "get_metadata"}', '192.168.1.102', 'Python-requests/2.31.0', NOW() - INTERVAL '1 day'),
    ('user_001', 'history_query', 'inference_history', NULL, '{"limit": 10, "model": "lstm_timeseries"}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', NOW() - INTERVAL '1 day' + INTERVAL '2 hours'),
    ('user_004', 'inference_submit', 'inference_job', 'job_12347', '{"model_name": "lstm_timeseries", "data_size": 10}', '192.168.1.103', 'curl/7.81.0', NOW() - INTERVAL '12 hours'),
    ('user_004', 'inference_complete', 'inference_job', 'job_12347', '{"status": "completed", "duration": 0.42}', '192.168.1.103', 'curl/7.81.0', NOW() - INTERVAL '12 hours' + INTERVAL '500 milliseconds'),
    ('user_002', 'stats_query', 'statistics', NULL, '{"type": "summary"}', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', NOW() - INTERVAL '6 hours'),
    ('user_005', 'inference_submit', 'inference_job', 'job_12348', '{"model_name": "moving_average", "data_size": 10}', '192.168.1.104', 'PostmanRuntime/7.32.3', NOW() - INTERVAL '3 hours'),
    ('user_005', 'inference_complete', 'inference_job', 'job_12348', '{"status": "completed", "duration": 0.02}', '192.168.1.104', 'PostmanRuntime/7.32.3', NOW() - INTERVAL '3 hours' + INTERVAL '150 milliseconds'),
    ('admin_001', 'system_check', 'health', NULL, '{"redis": "connected", "postgres": "connected"}', '192.168.1.10', 'curl/7.81.0', NOW() - INTERVAL '1 hour'),
    ('user_001', 'search_query', 'elasticsearch', NULL, '{"model_name": "lstm_timeseries", "limit": 5}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', NOW() - INTERVAL '30 minutes'),
    ('user_003', 'analytics_query', 'model_analytics', NULL, '{"type": "model_statistics"}', '192.168.1.102', 'Python-requests/2.31.0', NOW() - INTERVAL '15 minutes'),
    ('user_002', 'inference_submit', 'inference_job', 'job_12349', '{"model_name": "lstm_timeseries", "data_size": 10}', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', NOW() - INTERVAL '5 minutes');

-- Insert sample inference results for demo purposes
INSERT INTO inference_results (job_id, model_name, status, predictions, confidence, metrics, metadata, created_at, completed_at)
VALUES
    (
        'sample_job_001',
        'lstm_timeseries',
        'completed',
        '[28.5, 30.3, 32.1, 33.9, 35.7]',
        '{"overall": 0.945, "per_step": [0.98, 0.95, 0.93, 0.92, 0.90]}',
        '{"inference_time": 0.045, "model_version": "1.0.0"}',
        '{"submitted_by": "user_001", "request_id": "req_001"}',
        NOW() - INTERVAL '2 days',
        NOW() - INTERVAL '2 days' + INTERVAL '450 milliseconds'
    ),
    (
        'sample_job_002',
        'moving_average',
        'completed',
        '[20.5, 21.2, 22.0, 22.8, 23.5]',
        '{"overall": 0.878}',
        '{"inference_time": 0.002, "window_size": 5}',
        '{"submitted_by": "user_002", "request_id": "req_002"}',
        NOW() - INTERVAL '2 days' + INTERVAL '5 minutes',
        NOW() - INTERVAL '2 days' + INTERVAL '5 minutes' + INTERVAL '20 milliseconds'
    ),
    (
        'sample_job_003',
        'lstm_timeseries',
        'completed',
        '[165.5, 173.8, 182.5, 191.6, 201.2]',
        '{"overall": 0.938, "per_step": [0.96, 0.94, 0.93, 0.92, 0.91]}',
        '{"inference_time": 0.042, "model_version": "1.0.0"}',
        '{"submitted_by": "user_004", "request_id": "req_003"}',
        NOW() - INTERVAL '12 hours',
        NOW() - INTERVAL '12 hours' + INTERVAL '420 milliseconds'
    ),
    (
        'sample_job_004',
        'moving_average',
        'completed',
        '[145.5, 152.3, 159.5, 167.0, 174.8]',
        '{"overall": 0.891}',
        '{"inference_time": 0.002, "window_size": 3}',
        '{"submitted_by": "user_005", "request_id": "req_004"}',
        NOW() - INTERVAL '3 hours',
        NOW() - INTERVAL '3 hours' + INTERVAL '15 milliseconds'
    );
