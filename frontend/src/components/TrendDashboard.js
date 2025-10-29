import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { format, subHours } from 'date-fns';
import './TrendDashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api/v1';

const METRICS = [
  { key: 'satellite_temperature', label: 'Temperature', unit: '°C', color: '#4a9eff' },
  { key: 'satellite_altitude', label: 'Altitude', unit: 'km', color: '#a78bfa' },
  { key: 'satellite_velocity', label: 'Velocity', unit: 'km/s', color: '#fbbf24' },
  { key: 'satellite_battery_voltage', label: 'Battery', unit: 'V', color: '#4ade80' },
  { key: 'satellite_solar_power', label: 'Solar Power', unit: 'W', color: '#fb923c' }
];

// 메트릭별 최신 값을 저장하는 상태 추가 예정

const TIME_RANGES = [
  { label: '1m', hours: 1/60 },      // 1분
  { label: '5m', hours: 5/60 },      // 5분
  { label: '15m', hours: 15/60 },    // 15분
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '1d', hours: 24 },
];

function TrendDashboard() {
  const [selectedMetric, setSelectedMetric] = useState(METRICS[0]);
  const [selectedTimeRange, setSelectedTimeRange] = useState(TIME_RANGES[1]); // 5m (실시간 느낌)
  const [rawData, setRawData] = useState([]);
  const [predictionData, setPredictionData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [satellites, setSatellites] = useState([]);
  const [selectedSatellite, setSelectedSatellite] = useState('SAT-001');
  const [latestValues, setLatestValues] = useState({});
  const [latestPredictions, setLatestPredictions] = useState({});
  const [showRawData, setShowRawData] = useState(true);
  const [showPrediction, setShowPrediction] = useState(true);

  // 위성 목록 조회
  useEffect(() => {
    fetchSatellites();
  }, []);

  const fetchSatellites = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/trends/satellites`);
      const data = await response.json();
      if (data.satellites && data.satellites.length > 0) {
        setSatellites(data.satellites);
        setSelectedSatellite(data.satellites[0]);
      }
    } catch (err) {
      console.error('Failed to fetch satellites:', err);
    }
  };

  // 데이터 조회
  const fetchTrendData = useCallback(async () => {
    // 초기 로딩일 때만 loading 상태 설정 (재렌더링 최소화)
    if (rawData.length === 0) {
      setLoading(true);
    }
    setError(null);

    // 현재 시각 기준으로 시간 범위 설정 (UTC 기준)
    const now = Date.now(); // Unix timestamp (밀리초)
    const endTime = new Date(now);
    const startTime = new Date(now - selectedTimeRange.hours * 60 * 60 * 1000);

    try {
      // 원본 데이터 조회 (UTC 시간으로 쿼리)
      const rawParams = new URLSearchParams({
        metric: selectedMetric.key,
        start_time: startTime.toISOString(),
        end_time: endTime.toISOString(),
        satellite_id: selectedSatellite
      });

      const rawResponse = await fetch(`${API_BASE_URL}/trends/raw?${rawParams}`);
      if (!rawResponse.ok) throw new Error('Failed to fetch raw data');
      const rawResult = await rawResponse.json();

      // 데이터 포맷팅 (UTC 타임스탬프로 변환)
      const formattedRawData = rawResult.data_points.map(point => {
        // ISO 문자열이 Z 없이 오면 UTC로 처리
        const timestamp = point.timestamp.endsWith('Z')
          ? new Date(point.timestamp).getTime()
          : new Date(point.timestamp + 'Z').getTime();
        return {
          timestamp,
          value: point.value
        };
      });

      setRawData(formattedRawData);
      setStats(rawResult.summary);

      // 예측 데이터 조회 (VictoriaMetrics에서 직접) - 미래 5초 포함
      try {
        const predMetricName = `${selectedMetric.key}_prediction`;
        const predQuery = `${predMetricName}{satellite_id="${selectedSatellite}"}`;

        // Unix timestamp (초 단위) - 미래 5초 포함
        const startTimestamp = Math.floor((now - selectedTimeRange.hours * 60 * 60 * 1000) / 1000);
        const endTimestamp = Math.floor((now + 5000) / 1000);

        const predResponse = await fetch(
          `/victoriametrics/api/v1/query_range?query=${encodeURIComponent(predQuery)}&start=${startTimestamp}&end=${endTimestamp}&step=1s`
        );

        if (predResponse.ok) {
          const predResult = await predResponse.json();
          if (predResult.status === 'success' && predResult.data.result.length > 0) {
            const predValues = predResult.data.result[0].values || [];
            const formattedPredData = predValues.map(([timestamp, value]) => ({
              timestamp: timestamp * 1000,
              prediction: parseFloat(value)
            }));
            setPredictionData(formattedPredData);
          } else {
            setPredictionData([]);
          }
        }
      } catch (predErr) {
        console.warn('Prediction data not available:', predErr);
        setPredictionData([]);
      }

    } catch (err) {
      setError(err.message);
      console.error('Error fetching trend data:', err);
    } finally {
      if (rawData.length === 0) {
        setLoading(false);
      }
    }
  }, [selectedMetric, selectedTimeRange, selectedSatellite, rawData.length]);

  // 모든 메트릭의 최신 값을 가져오는 함수
  const fetchLatestValues = useCallback(async () => {
    try {
      // Raw 데이터 조회
      const rawPromises = METRICS.map(async (metric) => {
        const response = await fetch(
          `/victoriametrics/api/v1/query?query=${metric.key}{satellite_id="${selectedSatellite}"}`
        );
        const data = await response.json();

        if (data.status === 'success' && data.data.result.length > 0) {
          const value = parseFloat(data.data.result[0].value[1]);
          return { key: metric.key, value };
        }
        return { key: metric.key, value: null };
      });

      // 예측 데이터 조회
      const predictionPromises = METRICS.map(async (metric) => {
        const predMetric = `${metric.key}_prediction`;
        const response = await fetch(
          `/victoriametrics/api/v1/query?query=${predMetric}{satellite_id="${selectedSatellite}"}`
        );
        const data = await response.json();

        if (data.status === 'success' && data.data.result.length > 0) {
          const value = parseFloat(data.data.result[0].value[1]);
          return { key: metric.key, value };
        }
        return { key: metric.key, value: null };
      });

      const [rawResults, predResults] = await Promise.all([
        Promise.all(rawPromises),
        Promise.all(predictionPromises)
      ]);

      const valuesMap = {};
      rawResults.forEach(r => {
        valuesMap[r.key] = r.value;
      });
      setLatestValues(valuesMap);

      const predictionsMap = {};
      predResults.forEach(r => {
        predictionsMap[r.key] = r.value;
      });
      setLatestPredictions(predictionsMap);
    } catch (err) {
      console.error('Failed to fetch latest values:', err);
    }
  }, [selectedSatellite]);

  useEffect(() => {
    fetchTrendData();
    fetchLatestValues();

    // 자동 새로고침: 차트는 3초마다, 최신 값은 1초마다 업데이트
    const trendInterval = setInterval(fetchTrendData, 3000);
    const valuesInterval = setInterval(fetchLatestValues, 1000);

    return () => {
      clearInterval(trendInterval);
      clearInterval(valuesInterval);
    };
  }, [fetchTrendData, fetchLatestValues]);

  // 차트 데이터 병합
  const mergedChartData = React.useMemo(() => {
    const dataMap = new Map();

    rawData.forEach(point => {
      dataMap.set(point.timestamp, { timestamp: point.timestamp, raw: point.value });
    });

    predictionData.forEach(point => {
      const existing = dataMap.get(point.timestamp) || { timestamp: point.timestamp };
      existing.prediction = point.prediction;
      dataMap.set(point.timestamp, existing);
    });

    return Array.from(dataMap.values()).sort((a, b) => a.timestamp - b.timestamp);
  }, [rawData, predictionData]);

  // X축 도메인 계산 (고정 시간 윈도우, 최신 데이터가 우측에 표시)
  // 실제 데이터의 타임스탬프 범위를 기준으로 설정
  const xAxisDomain = React.useMemo(() => {
    if (rawData.length === 0 && predictionData.length === 0) {
      // 데이터가 없으면 기본 범위 반환
      const now = Date.now();
      return [now - selectedTimeRange.hours * 60 * 60 * 1000, now];
    }

    // 실제 데이터의 최소/최대 타임스탬프 계산
    const allTimestamps = [
      ...rawData.map(d => d.timestamp),
      ...predictionData.map(d => d.timestamp)
    ];

    if (allTimestamps.length === 0) {
      const now = Date.now();
      return [now - selectedTimeRange.hours * 60 * 60 * 1000, now];
    }

    const maxTimestamp = Math.max(...allTimestamps);

    // 시간 범위에 맞춰 윈도우 설정 (미래 포함하지 않음)
    const windowDuration = selectedTimeRange.hours * 60 * 60 * 1000;
    const windowStart = maxTimestamp - windowDuration;
    const windowEnd = maxTimestamp;

    return [windowStart, windowEnd];
  }, [selectedTimeRange, rawData, predictionData]);

  const formatXAxis = (timestamp) => {
    // 시간 범위에 따라 포맷 변경
    if (selectedTimeRange.hours <= 0.25) { // 15분 이하: 초 단위 표시
      return format(new Date(timestamp), 'HH:mm:ss');
    } else if (selectedTimeRange.hours <= 1) { // 1시간 이하: 분 단위 표시
      return format(new Date(timestamp), 'HH:mm');
    } else { // 그 이상: 시간 단위 표시
      return format(new Date(timestamp), 'HH:mm');
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-time">{format(new Date(label), 'yyyy-MM-dd HH:mm:ss')}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)} {selectedMetric.unit}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="trend-dashboard">
      {/* 헤더 */}
      <header className="dashboard-header">
        <div className="header-left">
          <h1>🛰️ Satellite Monitor</h1>
          {selectedSatellite && (
            <span className="satellite-badge">{selectedSatellite}</span>
          )}
        </div>

        <div className="header-controls">
          {/* 위성 선택 */}
          {satellites.length > 0 && (
            <select
              className="satellite-selector"
              value={selectedSatellite}
              onChange={(e) => setSelectedSatellite(e.target.value)}
            >
              {satellites.map(sat => (
                <option key={sat} value={sat}>{sat}</option>
              ))}
            </select>
          )}

          {/* 시간 범위 선택 */}
          {TIME_RANGES.map(range => (
            <button
              key={range.label}
              className={`time-range-btn ${selectedTimeRange.label === range.label ? 'active' : ''}`}
              onClick={() => setSelectedTimeRange(range)}
            >
              {range.label}
            </button>
          ))}

          <button className="refresh-btn" onClick={fetchTrendData} disabled={loading}>
            {loading ? '⟳' : '↻'}
          </button>
        </div>
      </header>

      {/* 메트릭 카드 */}
      <div className="metrics-grid">
        {METRICS.map(metric => {
          const latestValue = latestValues[metric.key];
          const predictionValue = latestPredictions[metric.key];
          const isSelected = selectedMetric.key === metric.key;

          return (
            <div
              key={metric.key}
              className={`metric-card ${isSelected ? 'selected' : ''}`}
              onClick={() => setSelectedMetric(metric)}
            >
              <div className="metric-label">{metric.label}</div>
              <div className="metric-stats">
                <div className="stat-value" style={{ color: metric.color }}>
                  {latestValue !== null && latestValue !== undefined
                    ? latestValue.toFixed(2)
                    : 'N/A'}{' '}
                  {metric.unit}
                </div>
                {predictionValue !== null && predictionValue !== undefined && (
                  <div className="stat-prediction" style={{ color: '#4ade80', fontSize: '0.85em', marginTop: '4px' }}>
                    Prediction: {predictionValue.toFixed(2)} {metric.unit}
                  </div>
                )}
                {isSelected && stats && (
                  <div className="stat-range">
                    Min: {stats.min?.toFixed(2) ?? 'N/A'} | Max: {stats.max?.toFixed(2) ?? 'N/A'}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* 에러 메시지 */}
      {error && (
        <div className="error-message">
          ⚠️ Error: {error}
        </div>
      )}

      {/* 메인 차트 */}
      <div className="chart-container">
        <div className="chart-header">
          <h2>{selectedMetric.label} Trend</h2>
          <div className="chart-legend">
            <span
              className={`legend-item ${showRawData ? 'active' : 'inactive'}`}
              onClick={() => setShowRawData(!showRawData)}
              style={{ cursor: 'pointer' }}
            >
              <span className="legend-line raw"></span>
              Raw Data
            </span>
            <span
              className={`legend-item ${showPrediction ? 'active' : 'inactive'}`}
              onClick={() => setShowPrediction(!showPrediction)}
              style={{ cursor: 'pointer' }}
            >
              <span className="legend-line prediction"></span>
              Prediction
            </span>
          </div>
        </div>

        {loading && mergedChartData.length === 0 ? (
          <div className="loading-state">Loading data...</div>
        ) : mergedChartData.length === 0 ? (
          <div className="empty-state">No data available</div>
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={mergedChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3f" />
              <XAxis
                dataKey="timestamp"
                type="number"
                domain={xAxisDomain}
                tickFormatter={formatXAxis}
                stroke="#a0a0a0"
                scale="time"
                tickCount={6}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#a0a0a0"
                label={{ value: selectedMetric.unit, angle: -90, position: 'insideLeft', fill: '#a0a0a0' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {showRawData && (
                <Line
                  type="monotone"
                  dataKey="raw"
                  stroke={selectedMetric.color}
                  strokeWidth={2}
                  dot={false}
                  name="Raw Data"
                  isAnimationActive={false}
                />
              )}
              {showPrediction && predictionData.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="prediction"
                  stroke="#4ade80"
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  dot={false}
                  name="Prediction"
                  isAnimationActive={false}
                  opacity={0.8}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* 통계 패널 */}
      {stats && (
        <div className="stats-panel">
          <h3>Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-label">Data Points</div>
              <div className="stat-value">{stats.count}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Mean</div>
              <div className="stat-value">{stats.mean?.toFixed(2) ?? 'N/A'} {selectedMetric.unit}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Std Dev</div>
              <div className="stat-value">{stats.std?.toFixed(2) ?? 'N/A'}</div>
            </div>
            <div className="stat-item">
              <div className="stat-label">Range</div>
              <div className="stat-value">
                {stats.min?.toFixed(2) ?? 'N/A'} ~ {stats.max?.toFixed(2) ?? 'N/A'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TrendDashboard;
